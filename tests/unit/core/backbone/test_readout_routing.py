"""Readout routing: readout-token option, candidate binding, task states, pools, head backward."""

import pytest
import torch
from torch import nn

from perseus.core.backbone import Backbone
from perseus.core.backbone.history import HistoryAggregator
from perseus.core.backbone.history.interaction.cross_attention import CrossAttention
from perseus.core.backbone.history.interaction.readout_attention import TargetAttentionPooling
from perseus.core.config import Config
from perseus.core.config import HistoryAggregator as HistoryConfig
from perseus.core.encoders.id import Embedder
from perseus.core.tasks.classification.target import Criterion, Layer

DIM = 32
EMB = 8
FIELDS = 3
MAX_EVENTS = 12
CONTEXT_FIELDS = ["user_id", "item", "brand"]


def _history(tokenize: dict | None = None, pool: dict | str | None = None, **query_params) -> dict:
    params = {"tokenize": tokenize if tokenize is not None else {"type": "field_task", "params": {"num_tasks": 4}}}
    params.update(query_params)
    return {
        "nonsequence_encoder": {"type": "stack", "params": {"num_layers": 2, "blocks": [
            {"type": "low_rank_attention", "params": {"num_heads": 2, "qkv_rank": 64}},
            {"type": "per_token_ffn", "params": {"expansion_factor": 2, "activation": "gelu"}},
        ]}},
        "interaction": {"schedule": "once", "type": "target_attention", "params": params,
                        "pool": pool if pool is not None else {"select": 4}},
    }


def _pool_only_history() -> dict:
    return {
        "nonsequence_encoder": {"type": "ffn", "params": {"num_layers": 2}, "pool": "mean"},
        "interaction": {"schedule": "once", "type": "target_attention",
                        "params": {}},
    }


def _backbone(*, readout: bool = True, num_tokens: int | None = None, history: dict | None = None,
              context_type: str = "identity", context_fields: list[str] | None = CONTEXT_FIELDS) -> Backbone:
    params = {} if num_tokens is None else {"num_tokens": num_tokens}
    config = None
    if context_type != "none":
        config = {"type": context_type, "num_features": FIELDS, "params": params}
    history = HistoryConfig.model_validate(history if history is not None else _history(
        query_context_fields=["item"],
    )).model_dump()
    return Backbone(
        DIM,
        history_aggregator_config={"max_events_per_sequence": MAX_EVENTS, **history},
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
        context_aggregator_config=config,
        embedding_dim=EMB,
        context_fields=context_fields,
        readout_token=readout,
    )


def _batch(batch: int = 2, cold: bool = False):
    generator = torch.Generator().manual_seed(13)
    events = {"event": torch.randint(1, 6, (batch, 5), generator=generator, dtype=torch.int32)}
    positions = torch.zeros(batch, 5, dtype=torch.int32) if cold else torch.ones(batch, 5, dtype=torch.int32)
    timestamps = torch.arange(5, dtype=torch.float32).unsqueeze(0).repeat(batch, 1) + 1.0
    sample = torch.full((batch,), 7.0)
    context = {name: torch.randint(1, 6, (batch,), generator=generator, dtype=torch.int32)
               for name in CONTEXT_FIELDS}
    embedders = {"event": Embedder(6, EMB)} | {name: Embedder(6, EMB) for name in CONTEXT_FIELDS}
    return embedders, events, positions, timestamps, sample, context


class TestReadoutTokenOption:
    def test_readout_present_by_default_and_absent_when_disabled(self) -> None:
        assert isinstance(_backbone().readout_token, nn.Parameter)
        assert _backbone(readout=False).readout_token is None

    def test_parameter_count_shrinks_by_one_token(self) -> None:
        history = _pool_only_history()
        history["interaction"]["params"]["query_context_fields"] = ["item"]
        with_readout = sum(p.numel() for p in _backbone(history=history).parameters())
        without = sum(p.numel() for p in _backbone(history=history, readout=False).parameters())
        assert with_readout - without == DIM

    def test_reader_width_drops_the_readout(self) -> None:
        history = {
            "nonsequence_encoder": {"type": "ffn", "params": {"num_layers": 2}, "pool": "flatten"},
            "interaction": {"schedule": "layered", "type": "cross_attention",
                            "params": {"projection": "per_token", "num_heads": 2}},
        }
        attention_on = _backbone(history=history).history_aggregator.interaction.layers[0]
        attention_off = _backbone(history=history, readout=False).history_aggregator.interaction.layers[0]
        assert isinstance(attention_on, CrossAttention)
        assert isinstance(attention_off, CrossAttention)
        assert attention_on.query_weight.shape[0] == FIELDS + 1
        assert attention_off.query_weight.shape[0] == FIELDS

    def test_pool_readout_requires_the_token(self) -> None:
        history = _history(query_context_fields=["item"])
        history["interaction"]["pool"] = "readout"
        with pytest.raises(ValueError, match="readout pool requires the backbone readout token"):
            _backbone(readout=False, history=history)

    def test_target_attention_requires_a_query_without_readout(self) -> None:
        history = {
            "nonsequence_encoder": {"type": "ffn", "params": {"num_layers": 2}, "pool": "mean"},
            "interaction": {"schedule": "once", "type": "target_attention",
                            "params": {}},
        }
        with pytest.raises(ValueError, match="requires a readout token or selected context query"):
            _backbone(readout=False, history=history)
        history["interaction"]["params"]["query_context_fields"] = ["item"]
        backbone = _backbone(readout=False, history=history)
        assert backbone.output_shape == (DIM,)

    def test_cross_attention_context_reader_requires_a_stream(self) -> None:
        history = {
            "nonsequence_encoder": {"type": "ffn", "params": {"num_layers": 2}, "pool": "mean"},
            "interaction": {"schedule": "layered", "type": "cross_attention",
                            "params": {"projection": "per_token", "num_heads": 2}},
        }
        with pytest.raises(ValueError, match="requires context or a readout token"):
            _backbone(readout=False, context_type="none", history=history)

    def test_mask_has_no_readout_column_when_disabled(self) -> None:
        backbone = _backbone(readout=False)
        events_positions = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.int32)
        mask = backbone._build_mask(events_positions, FIELDS)
        assert mask.shape == (1, 5 + FIELDS, 5 + FIELDS)

    def test_mask_with_readout_keeps_legacy_width(self) -> None:
        backbone = _backbone()
        events_positions = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.int32)
        mask = backbone._build_mask(events_positions, FIELDS)
        assert mask.shape == (1, 5 + FIELDS + 1, 5 + FIELDS + 1)

    def test_forward_without_any_context_block(self) -> None:
        history = {"sequence_encoder": {"type": "ffn", "params": {"num_layers": 2}, "pool": "mean"},
                   "interaction": {"schedule": "layered", "type": "identity"}}
        backbone = _backbone(readout=False, context_type="none", history=history).eval()
        embedders, events, positions, timestamps, sample, _ = _batch()
        out = backbone(embedders, events, positions, timestamps, sample)
        assert out.shape == (2, DIM)
        assert torch.isfinite(out).all()

    def test_pool_on_empty_context_group_is_rejected(self) -> None:
        history = {"nonsequence_encoder": {"type": "ffn", "params": {"num_layers": 2}, "pool": "mean"},
                   "interaction": {"schedule": "layered", "type": "identity"}}
        with pytest.raises(ValueError, match="is empty: add context features or the readout token"):
            _backbone(readout=False, context_type="none", history=history)


class TestCandidateBinding:
    def test_fields_resolve_to_context_token_indices(self) -> None:
        backbone = _backbone(history=_history(query_context_fields=["item", "brand"]))
        compress = backbone.history_aggregator.interaction.pools[0]
        assert isinstance(compress, TargetAttentionPooling)
        assert compress.query_context_indices == (1, 2)

    def test_indices_selector_still_works(self) -> None:
        backbone = _backbone(history=_history(query_context_indices=[2]))
        compress = backbone.history_aggregator.interaction.pools[0]
        assert compress.query_context_indices == (2,)

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown context fields"):
            _backbone(history=_history(query_context_fields=["target"]))

    def test_duplicate_fields_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            _backbone(history=_history(query_context_fields=["item", "item"]))

    def test_fields_and_indices_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            _backbone(history=_history(query_context_fields=["item"], query_context_indices=[1]))

    def test_fields_require_field_aligned_tokens(self) -> None:
        with pytest.raises(ValueError, match="field-aligned"):
            _backbone(context_type="tokenizer", num_tokens=4,
                      history=_history(query_context_fields=["item"]))

    def test_fields_resolve_on_fieldwise_tokenizer(self) -> None:
        backbone = _backbone(context_type="tokenizer",
                             history=_history(query_context_fields=["item", "user_id"]))
        compress = backbone.history_aggregator.interaction.pools[0]
        assert compress.query_context_indices == (1, 0)

    def test_pooling_query_is_the_sum_of_selected_tokens(self) -> None:
        pooling = TargetAttentionPooling(DIM, 0.0, query_context_indices=(1, 2))
        generator = torch.Generator().manual_seed(17)
        history = torch.randn(2, 6, DIM, generator=generator)
        group = torch.randn(2, FIELDS + 1, DIM, generator=generator)
        padding = torch.zeros(2, 6, dtype=torch.bool)
        out = pooling(history, group, padding)
        expected = pooling.pooling(group[:, 1].add(group[:, 2]), history, padding)
        torch.testing.assert_close(out, expected)


class TestTaskStates:
    def test_hiformer_emits_task_states_in_order(self) -> None:
        backbone = _backbone().eval()
        assert backbone.output_shape == (4, DIM)
        embedders, events, positions, timestamps, sample, context = _batch()
        states = backbone(embedders, events, positions, timestamps, sample, context)
        assert states.shape == (2, 4, DIM)

    def test_five_targets_emit_five_states(self) -> None:
        backbone = _backbone(history=_history(tokenize={"type": "field_task", "params": {"num_tasks": 5}},
                                              pool={"select": 5})).eval()
        assert backbone.output_shape == (5, DIM)
        embedders, events, positions, timestamps, sample, context = _batch()
        states = backbone(embedders, events, positions, timestamps, sample, context)
        assert states.shape == (2, 5, DIM)

    def test_target_head_binds_to_its_own_state(self) -> None:
        generator = torch.Generator().manual_seed(19)
        states = torch.randn(3, 4, DIM, generator=generator)
        layer = Layer(DIM, [2] * 4)
        output = layer(states, None)
        assert [part.shape for part in output] == [(3, 1)] * 4
        for index in range(4):
            torch.testing.assert_close(output[index], layer.layers[index](states[:, index]))

    def test_state_order_follows_label_order(self) -> None:
        generator = torch.Generator().manual_seed(23)
        states = torch.randn(2, 4, DIM, generator=generator)
        layer = Layer(DIM, [2] * 4)
        permutation = [2, 0, 3, 1]
        permuted = layer(states[:, permutation], None)
        for column in range(4):
            expected = layer.layers[column](states[:, permutation[column]])
            torch.testing.assert_close(permuted[column], expected)

    def test_task_tokens_survive_readout_disabled(self) -> None:
        backbone = _backbone(readout=False,
                             history=_history(query_context_fields=["item", "brand"])).eval()
        assert backbone.output_shape == (4, DIM)
        embedders, events, positions, timestamps, sample, context = _batch()
        states = backbone(embedders, events, positions, timestamps, sample, context)
        assert states.shape == (2, 4, DIM)

    def test_tokenmixer_has_no_task_tokens(self) -> None:
        history = {
            "nonsequence_encoder": {"type": "stack", "params": {"num_layers": 2, "blocks": [
                {"type": "token_mixing", "params": {"kernel": "permutation"}},
                {"type": "per_token_ffn", "params": {"expansion_factor": 2, "activation": "swiglu"}},
            ]}},
            "interaction": {"schedule": "once", "type": "target_attention",
                            "params": {"query_context_fields": ["item"],
                                       "tokenize": {"type": "global_group", "params": {"num_group_tokens": 4}}},
                            "pool": "mean"},
        }
        model = HistoryAggregator(
            {"max_events_per_sequence": MAX_EVENTS, **history}, DIM, FIELDS,
            context_fields=CONTEXT_FIELDS,
        )
        assert model.output_shape == (DIM,)
        tokens_module = model.interaction.tokenizers[0]
        assert tokens_module.group_tokenizer.num_tokens == 4


class TestReferencePools:
    def test_est_dual_pool_widths(self) -> None:
        history = {
            "num_layers": 2,
            "sequence_encoder": {"type": "ffn", "pool": "mean"},
            "nonsequence_encoder": {"type": "ffn", "pool": "flatten"},
            "interaction": {"schedule": "layered", "type": "cross_attention",
                            "params": {"projection": "per_token", "num_heads": 2}},
        }
        on = HistoryAggregator({"max_events_per_sequence": MAX_EVENTS, **history}, DIM, FIELDS)
        off = HistoryAggregator({"max_events_per_sequence": MAX_EVENTS, **history}, DIM, FIELDS, readout=False)
        assert on.output_shape == (DIM + (FIELDS + 1) * DIM,)
        assert off.output_shape == (DIM + FIELDS * DIM,)

    def test_queries_mean_pool_width(self) -> None:
        history = {
            "num_layers": 2,
            "sequence_encoder": {"type": "self_attention",
                                 "params": {"num_heads": 2,
                                            "ffn": {"expansion_factor": 2, "activation": "relu"}}},
            "nonsequence_encoder": {"type": "token_mixer",
                                    "params": {"kernel": "permutation",
                                               "queries": {"type": "persistent", "params": {"num_tokens": 2}},
                                               "ffn": {"expansion_factor": 2, "activation": "gelu"}}},
            "interaction": {"schedule": "layered", "type": "cross_attention",
                            "params": {"reader": "queries", "num_tokens": 2, "projection": "shared",
                                       "num_heads": 2, "write": "replace"}, "pool": "mean"},
        }
        model = HistoryAggregator({"max_events_per_sequence": MAX_EVENTS, **history}, DIM, FIELDS)
        assert model.output_shape == (DIM,)

    def test_flatten_pool_on_fieldwise_tokenizer(self) -> None:
        history = _history(pool="flatten")
        model = HistoryAggregator({"max_events_per_sequence": MAX_EVENTS, **history}, DIM, FIELDS,
                              context_fields=CONTEXT_FIELDS)
        assert model.output_shape == ((FIELDS + 2 + 4) * DIM,)


class TestHeadBackward:
    @pytest.mark.parametrize("cold", [False, True])
    @pytest.mark.parametrize("hidden", [None, [16, 8]])
    def test_bce_backward_through_real_heads(self, cold: bool, hidden) -> None:
        backbone = _backbone().eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        layer = Layer(DIM, [2] * 4, hidden=hidden)
        embedders, events, positions, timestamps, sample, context = _batch(cold=cold)
        states = backbone(embedders, events, positions, timestamps, sample, context)
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
        loss = Criterion(layer)(states, None, target)
        loss.backward()
        assert torch.isfinite(loss)
        for target_layer in layer.layers:
            for parameter in target_layer.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
        for parameter in backbone.context_aggregator.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_task_state_backward_binds_each_head(self) -> None:
        backbone = _backbone(readout=False,
                             history=_history(query_context_fields=["item", "brand"])).eval()
        layer = Layer(DIM, [2] * 4)
        embedders, events, positions, timestamps, sample, context = _batch()
        states = backbone(embedders, events, positions, timestamps, sample, context)
        target = torch.ones(2, 4)
        Criterion(layer)(states, None, target).backward()
        for target_layer in layer.layers:
            grad = target_layer.weight.grad
            assert grad is not None
            assert torch.isfinite(grad).all()


class TestConfigSchema:
    def test_readout_token_flag_flows_through_config(self) -> None:
        config_dict = {
            "task": {"type": "classification", "metrics": {"logloss": None}},
            "events": {"purchase": {"attributes": {"item": None}}},
            "embedding_dim": EMB,
            "encoders": {
                "item": {"type": "id", "preprocessor": {"min_count": 1}},
                "user": {"type": "id", "preprocessor": {"min_count": 1}},
            },
            "features": {
                "item": {"located_in": {"event": True, "context": True}, "encoder": "item"},
                "user": {"located_in": {"context": True}, "encoder": "user"},
            },
            "backbone": {
                "dim": DIM,
                "readout_token": False,
                "event_aggregator": {"type": "sum"},
                "context_aggregator": {"type": "tokenizer", "params": {}},
                "history_aggregator": {
                    "nonsequence_encoder": {"type": "stack", "params": {"num_layers": 2, "blocks": [
                        {"type": "token_mixing", "params": {"kernel": "permutation"}},
                        {"type": "per_token_ffn", "params": {"expansion_factor": 2, "activation": "swiglu"}},
                    ]}},
                    "interaction": {"schedule": "once", "type": "target_attention",
                                    "params": {"query_context_fields": ["item"],
                                               "tokenize": {"type": "global_group", "params": {"num_group_tokens": 4}}},
                                    "pool": "mean"},
                },
            },
        }
        config = Config.model_validate(config_dict)
        assert config.backbone.readout_token is False
        context_fields = list(config.feature_to_encoder(context=True))
        assert context_fields == ["item", "user"]
        backbone = Backbone(
            config.backbone.dim,
            history_aggregator_config={
                "max_events_per_sequence": config.max_events_per_sequence,
                **config.backbone.history_aggregator.model_dump(),
            },
            event_aggregator_config={
                "num_features": len(config.feature_to_encoder(event=True)),
                **config.backbone.event_aggregator.model_dump(),
            },
            context_aggregator_config={
                "num_features": len(context_fields),
                **config.backbone.context_aggregator.model_dump(),
            },
            embedding_dim=config.embedding_dim,
            context_fields=context_fields,
            readout_token=config.backbone.readout_token,
        ).eval()
        assert backbone.readout_token is None
        assert backbone.output_shape == (DIM,)
        assert backbone.history_aggregator.interaction.pools[0].query_context_indices == (0,)
