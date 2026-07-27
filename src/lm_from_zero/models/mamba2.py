"""Project-owned Mamba-2 state-space causal language model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lm_from_zero.models.config import Mamba2Config
from lm_from_zero.models.interfaces import (
    CausalLMOutput,
    Mamba2Cache,
    Mamba2LayerState,
)


def _validate_ssd_inputs(
    x: Tensor,
    log_decay: Tensor,
    b: Tensor,
    c: Tensor,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if x.ndim != 4:
        raise ValueError("x must have shape [batch, sequence, heads, head_dim]")
    batch, sequence, heads, head_dim = x.shape
    if log_decay.shape != (batch, sequence, heads):
        raise ValueError("log_decay must match the x batch, sequence, and heads")
    if b.ndim != 4 or c.shape != b.shape or b.shape[:2] != (batch, sequence):
        raise ValueError("B and C must share [batch, sequence, groups, state]")
    groups = b.shape[2]
    state_size = b.shape[3]
    if groups == 0 or heads % groups != 0:
        raise ValueError("SSD heads must be divisible by B/C groups")
    b_heads = b.repeat_interleave(heads // groups, dim=2)
    c_heads = c.repeat_interleave(heads // groups, dim=2)
    expected_state = (batch, heads, head_dim, state_size)
    if initial_state is None:
        state = torch.zeros(expected_state, dtype=x.dtype, device=x.device)
    else:
        if initial_state.shape != expected_state:
            raise ValueError("initial SSM state has an incompatible shape")
        state = initial_state.to(dtype=x.dtype, device=x.device)
    return b_heads, c_heads, state


def _segment_sum(values: Tensor) -> Tensor:
    """Return stable lower-triangular segment sums along the last axis."""

    length = values.shape[-1]
    expanded = values.unsqueeze(-1).expand(*values.shape, length)
    strict_lower = torch.ones(
        length,
        length,
        dtype=torch.bool,
        device=values.device,
    ).tril(diagonal=-1)
    segments = expanded.masked_fill(~strict_lower, 0).cumsum(dim=-2)
    lower = torch.ones(
        length,
        length,
        dtype=torch.bool,
        device=values.device,
    ).tril()
    return segments.masked_fill(~lower, -torch.inf)


def ssd_sequential_reference(
    x: Tensor,
    log_decay: Tensor,
    b: Tensor,
    c: Tensor,
    *,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate the discrete SSD recurrence one token at a time."""

    b_heads, c_heads, state = _validate_ssd_inputs(
        x,
        log_decay,
        b,
        c,
        initial_state,
    )
    outputs: list[Tensor] = []
    for index in range(x.shape[1]):
        decay = log_decay[:, index].exp().unsqueeze(-1).unsqueeze(-1)
        contribution = torch.einsum(
            "bhn,bhp->bhpn",
            b_heads[:, index],
            x[:, index],
        )
        state = decay * state + contribution
        outputs.append(
            torch.einsum(
                "bhpn,bhn->bhp",
                state,
                c_heads[:, index],
            )
        )
    return torch.stack(outputs, dim=1), state


def ssd_quadratic_reference(
    x: Tensor,
    log_decay: Tensor,
    b: Tensor,
    c: Tensor,
) -> Tensor:
    """Evaluate SSD through its explicit causal semiseparable matrix."""

    b_heads, c_heads, _ = _validate_ssd_inputs(x, log_decay, b, c, None)
    decay_matrix = _segment_sum(log_decay.transpose(1, 2)).exp()
    return torch.einsum(
        "bihn,bjhn,bhij,bjhp->bihp",
        c_heads,
        b_heads,
        decay_matrix,
        x,
    )


def _pad_sequence(values: Tensor, padded_length: int) -> Tensor:
    padding = padded_length - values.shape[1]
    if padding == 0:
        return values
    zeros = torch.zeros(
        values.shape[0],
        padding,
        *values.shape[2:],
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat((values, zeros), dim=1)


def ssd_chunked(
    x: Tensor,
    log_decay: Tensor,
    b: Tensor,
    c: Tensor,
    *,
    chunk_size: int,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate SSD with the four-part chunk decomposition from Mamba-2."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    b_heads, c_heads, state = _validate_ssd_inputs(
        x,
        log_decay,
        b,
        c,
        initial_state,
    )
    batch, sequence, heads, head_dim = x.shape
    state_size = b_heads.shape[-1]
    padded_length = math.ceil(sequence / chunk_size) * chunk_size
    chunks = padded_length // chunk_size
    x_padded = _pad_sequence(x, padded_length)
    a_padded = _pad_sequence(log_decay, padded_length)
    b_padded = _pad_sequence(b_heads, padded_length)
    c_padded = _pad_sequence(c_heads, padded_length)

    x_chunks = x_padded.view(batch, chunks, chunk_size, heads, head_dim)
    b_chunks = b_padded.view(batch, chunks, chunk_size, heads, state_size)
    c_chunks = c_padded.view(batch, chunks, chunk_size, heads, state_size)
    a_chunks = (
        a_padded.view(batch, chunks, chunk_size, heads).permute(0, 3, 1, 2).contiguous()
    )
    a_cumulative = a_chunks.cumsum(dim=-1)

    diagonal_decay = _segment_sum(a_chunks).exp()
    diagonal_output = torch.einsum(
        "bcqhn,bckhn,bhcqk,bckhp->bcqhp",
        c_chunks,
        b_chunks,
        diagonal_decay,
        x_chunks,
    )

    state_decay = (a_cumulative[..., -1:] - a_cumulative).exp()
    chunk_states = torch.einsum(
        "bcqhn,bhcq,bcqhp->bchpn",
        b_chunks,
        state_decay,
        x_chunks,
    )

    state_inputs = torch.cat((state.unsqueeze(1), chunk_states), dim=1)
    chunk_decay = _segment_sum(F.pad(a_cumulative[..., -1], (1, 0))).exp()
    boundary_states = torch.einsum(
        "bhzc,bchpn->bzhpn",
        chunk_decay,
        state_inputs,
    )
    start_states = boundary_states[:, :-1]
    final_state = boundary_states[:, -1]

    off_diagonal_output = torch.einsum(
        "bcqhn,bchpn,bhcq->bcqhp",
        c_chunks,
        start_states,
        a_cumulative.exp(),
    )
    output = (diagonal_output + off_diagonal_output).reshape(
        batch,
        padded_length,
        heads,
        head_dim,
    )
    return output[:, :sequence], final_state


class MambaRMSNorm(nn.Module):
    """RMS normalization with fp32 variance accumulation."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        normalized = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(input_dtype)


class GroupedGatedRMSNorm(nn.Module):
    """Official-style group RMSNorm applied after SiLU gating."""

    def __init__(self, hidden_size: int, groups: int, eps: float) -> None:
        super().__init__()
        if hidden_size % groups != 0:
            raise ValueError("gated RMSNorm width must be divisible by groups")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.groups = groups
        self.group_size = hidden_size // groups
        self.eps = eps

    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        gated = hidden_states.float() * F.silu(gate.float())
        grouped = gated.view(*gated.shape[:-1], self.groups, self.group_size)
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(dim=-1, keepdim=True) + self.eps
        )
        flattened = normalized.flatten(start_dim=-2)
        return (flattened * self.weight.float()).to(input_dtype)


class Mamba2Mixer(nn.Module):
    """Causal convolution followed by grouped selective SSD."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        projection_size = (
            2 * config.inner_size
            + 2 * config.num_groups * config.state_size
            + config.num_heads
        )
        self.in_proj = nn.Linear(
            config.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        self.conv1d = nn.Conv1d(
            config.convolution_size,
            config.convolution_size,
            kernel_size=config.conv_kernel,
            groups=config.convolution_size,
            padding=config.conv_kernel - 1,
            bias=config.use_conv_bias,
        )
        dt = torch.exp(
            torch.rand(config.num_heads)
            * (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        ).clamp_min(config.time_step_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        initialized_a = torch.empty(config.num_heads, dtype=torch.float32).uniform_(
            config.a_init_min,
            config.a_init_max,
        )
        self.A_log = nn.Parameter(initialized_a.log())
        self.D = nn.Parameter(torch.ones(config.num_heads))
        self.norm = GroupedGatedRMSNorm(
            config.inner_size,
            config.num_groups,
            config.rms_norm_eps,
        )
        self.out_proj = nn.Linear(
            config.inner_size,
            config.hidden_size,
            bias=config.use_bias,
        )

    def _split_projection(self, projected: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        parts = torch.split(
            projected,
            [
                self.config.inner_size,
                self.config.convolution_size,
                self.config.num_heads,
            ],
            dim=-1,
        )
        return parts[0], parts[1], parts[2]

    def _split_convolution(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        grouped_state = self.config.num_groups * self.config.state_size
        parts = torch.split(
            values,
            [self.config.inner_size, grouped_state, grouped_state],
            dim=-1,
        )
        return parts[0], parts[1], parts[2]

    def _convolution_state(self, raw_xbc: Tensor) -> Tensor:
        transposed = raw_xbc.transpose(1, 2)
        if transposed.shape[-1] < self.config.conv_kernel:
            transposed = F.pad(
                transposed,
                (self.config.conv_kernel - transposed.shape[-1], 0),
            )
        return transposed[..., -self.config.conv_kernel :]

    def _full_forward(
        self,
        projected: Tensor,
    ) -> tuple[Tensor, Mamba2LayerState]:
        batch, sequence, _ = projected.shape
        z, raw_xbc, raw_dt = self._split_projection(projected)
        convolved = self.conv1d(raw_xbc.transpose(1, 2)).transpose(1, 2)
        convolved = F.silu(convolved[:, :sequence])
        x, b, c = self._split_convolution(convolved)
        dt = F.softplus(raw_dt.float() + self.dt_bias.float())
        a = -torch.exp(self.A_log.float())
        x_heads = x.float().view(
            batch,
            sequence,
            self.config.num_heads,
            self.config.head_dim,
        )
        b_grouped = b.float().view(
            batch,
            sequence,
            self.config.num_groups,
            self.config.state_size,
        )
        c_grouped = c.float().view(
            batch,
            sequence,
            self.config.num_groups,
            self.config.state_size,
        )
        y, final_state = ssd_chunked(
            x_heads * dt.unsqueeze(-1),
            dt * a.view(1, 1, -1),
            b_grouped,
            c_grouped,
            chunk_size=self.config.chunk_size,
        )
        y = y + self.D.float().view(1, 1, -1, 1) * x_heads
        flattened = y.flatten(start_dim=-2)
        output = self.out_proj(self.norm(flattened, z))
        state = Mamba2LayerState(
            convolution=self._convolution_state(raw_xbc),
            ssm=final_state,
        )
        return output, state

    def _validate_state(self, hidden_states: Tensor, state: Mamba2LayerState) -> None:
        batch = hidden_states.shape[0]
        expected_convolution = (
            batch,
            self.config.convolution_size,
            self.config.conv_kernel,
        )
        expected_ssm = (
            batch,
            self.config.num_heads,
            self.config.head_dim,
            self.config.state_size,
        )
        if state.convolution.shape != expected_convolution:
            raise ValueError("cached convolution state has an incompatible shape")
        if state.ssm.shape != expected_ssm:
            raise ValueError("cached SSM state has an incompatible shape")

    def _step(
        self,
        projected: Tensor,
        state: Mamba2LayerState,
        active: Tensor | None,
    ) -> tuple[Tensor, Mamba2LayerState]:
        z, raw_xbc, raw_dt = self._split_projection(projected)
        convolution = torch.cat(
            (state.convolution[..., 1:], raw_xbc.unsqueeze(-1)),
            dim=-1,
        )
        convolved = torch.einsum(
            "bdk,dk->bd",
            convolution,
            self.conv1d.weight[:, 0],
        )
        if self.conv1d.bias is not None:
            convolved = convolved + self.conv1d.bias
        x, b, c = self._split_convolution(F.silu(convolved))
        batch = projected.shape[0]
        x_heads = x.float().view(
            batch,
            self.config.num_heads,
            self.config.head_dim,
        )
        b_heads = (
            b.float()
            .view(batch, self.config.num_groups, self.config.state_size)
            .repeat_interleave(self.config.heads_per_group, dim=1)
        )
        c_heads = (
            c.float()
            .view(batch, self.config.num_groups, self.config.state_size)
            .repeat_interleave(self.config.heads_per_group, dim=1)
        )
        dt = F.softplus(raw_dt.float() + self.dt_bias.float())
        a = -torch.exp(self.A_log.float())
        decay = torch.exp(dt * a).unsqueeze(-1).unsqueeze(-1)
        contribution = torch.einsum(
            "bh,bhn,bhp->bhpn",
            dt,
            b_heads,
            x_heads,
        )
        ssm = state.ssm.float() * decay + contribution
        if active is not None:
            state_mask = active.view(batch, 1, 1, 1)
            convolution = torch.where(
                active.view(batch, 1, 1),
                convolution,
                state.convolution,
            )
            ssm = torch.where(state_mask, ssm, state.ssm.float())
        y = torch.einsum("bhpn,bhn->bhp", ssm, c_heads)
        y = y + self.D.float().view(1, -1, 1) * x_heads
        flattened = y.flatten(start_dim=-2)
        output = self.out_proj(self.norm(flattened, z))
        if active is not None:
            output = torch.where(active.view(batch, 1), output, 0)
        return output, Mamba2LayerState(convolution=convolution, ssm=ssm)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        state: Mamba2LayerState | None = None,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Mamba2LayerState]:
        if hidden_states.ndim != 3:
            raise ValueError("mixer input must have shape [batch, sequence, hidden]")
        if active_mask is not None and active_mask.shape != hidden_states.shape[:2]:
            raise ValueError("active mask must match mixer batch and sequence")
        projected = self.in_proj(hidden_states)
        if state is None and active_mask is None:
            return self._full_forward(projected)
        if state is None:
            state = Mamba2LayerState(
                convolution=torch.zeros(
                    hidden_states.shape[0],
                    self.config.convolution_size,
                    self.config.conv_kernel,
                    dtype=projected.dtype,
                    device=projected.device,
                ),
                ssm=torch.zeros(
                    hidden_states.shape[0],
                    self.config.num_heads,
                    self.config.head_dim,
                    self.config.state_size,
                    dtype=torch.float32,
                    device=projected.device,
                ),
            )
        else:
            self._validate_state(hidden_states, state)
        outputs: list[Tensor] = []
        current = state
        for index in range(hidden_states.shape[1]):
            active = None if active_mask is None else active_mask[:, index]
            output, current = self._step(projected[:, index], current, active)
            outputs.append(output)
        return torch.stack(outputs, dim=1), current


class Mamba2Block(nn.Module):
    """Pre-normalized Mamba-2 mixer with fp32 residual accumulation."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.norm = MambaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mixer = Mamba2Mixer(config)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        state: Mamba2LayerState | None = None,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Mamba2LayerState]:
        residual = hidden_states.float()
        mixed, new_state = self.mixer(
            self.norm(hidden_states),
            state=state,
            active_mask=active_mask,
        )
        return residual + mixed.float(), new_state


def _causal_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("labels must match input batch and sequence dimensions")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long token IDs")
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_labels.numel() == 0 or not torch.any(shift_labels != -100):
        return shift_logits.sum() * 0.0
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class Mamba2ForCausalLM(nn.Module):
    """Untied project-owned Mamba-2 causal language model."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            Mamba2Block(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = MambaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_module)
        with torch.no_grad():
            self.embed_tokens.weight[config.pad_token_id].zero_()

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, MambaRMSNorm | GroupedGatedRMSNorm):
            nn.init.ones_(module.weight)

    def _validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        cache: Mamba2Cache | None,
    ) -> Tensor | None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs")
        if input_ids.shape[1] == 0:
            raise ValueError("input sequence cannot be empty")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the vocabulary")
        if cache is not None and len(cache.layers) != len(self.layers):
            raise ValueError("cache layer count does not match the model")
        past_length = 0 if cache is None else cache.sequence_length
        total_length = past_length + input_ids.shape[1]
        if total_length > self.config.max_position_embeddings:
            raise ValueError("input and cache exceed the configured context length")
        if position_ids is not None:
            if (
                position_ids.shape != input_ids.shape
                or position_ids.dtype != torch.long
            ):
                raise ValueError("position_ids must match input_ids with torch.long")
            if torch.any(position_ids < 0) or torch.any(
                position_ids >= self.config.max_position_embeddings
            ):
                raise ValueError("position_ids are outside the configured context")
        if attention_mask is None:
            return None
        if attention_mask.shape != (input_ids.shape[0], total_length):
            raise ValueError(
                "attention_mask must cover the batch and complete token history"
            )
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask values must be zero or one")
        boolean_mask = attention_mask.to(dtype=torch.bool)
        if cache is None:
            if not torch.all(boolean_mask.any(dim=1)):
                raise ValueError("each sequence must contain at least one active token")
            right_padding = boolean_mask[:, :-1] & ~boolean_mask[:, 1:]
            if torch.any(right_padding):
                raise ValueError("Mamba-2 supports left padding but not right padding")
        current = boolean_mask[:, -input_ids.shape[1] :]
        return None if bool(torch.all(current)) else current

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: Mamba2Cache | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """Run causal logits, optional shifted loss, and recurrent caching."""

        active_mask = self._validate_inputs(
            input_ids,
            attention_mask,
            position_ids,
            cache,
        )
        hidden_states = self.embed_tokens(input_ids)
        new_states: list[Mamba2LayerState] = []
        for index, layer in enumerate(self.layers):
            layer_state = None if cache is None else cache.layers[index]
            hidden_states, new_state = layer(
                hidden_states,
                state=layer_state,
                active_mask=active_mask,
            )
            new_states.append(new_state)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None if labels is None else _causal_loss(logits, labels)
        output_cache = None
        if use_cache:
            previous_length = 0 if cache is None else cache.sequence_length
            output_cache = Mamba2Cache(
                layers=tuple(new_states),
                sequence_length=previous_length + input_ids.shape[1],
            )
        return CausalLMOutput(logits=logits, loss=loss, cache=output_cache)

    def trainable_parameter_count(self) -> int:
        """Return the authoritative realized trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())
