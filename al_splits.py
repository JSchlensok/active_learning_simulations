"""Extrapolation splits: what the surrogate model starts out knowing, and what it must reach.

A **split** partitions a dataset's labelled variants into a *train* set the campaign starts from
and a *test* set it has to get to. Each split poses one generalization question.
These splits occur under different names in the literature:

| Axis (here)               | METL (Gelman et al. 2025) | FLIP2 (Didi et al. 2026) | CombinGym (Chen et al. 2026) |
| ------------------------- | ------------------------- | ------------------------ | ---------------------------- |
| :attr:`~ExtrapolationAxis.NUMBER`    | regime extrapolation   | number    | *n*-vs-rest |
| :attr:`~ExtrapolationAxis.POSITION`  | position extrapolation | position  | — |
| :attr:`~ExtrapolationAxis.MUTATION`  | mutation extrapolation | mutation  | — |
| :attr:`~ExtrapolationAxis.SCORE`     | score extrapolation    | fitness / low-to-high | — |
| :attr:`~ExtrapolationAxis.WILDTYPE`  | —                      | wild type | — |

**How this differs from the supervised setting.** In METL and FLIP2 the test half is never trained
on. Here the campaign *may* label test variants as it goes — that is the point of active learning.
The train half is the campaign's starting knowledge (materialized as ``start_ids``), and the test
half stays in the pool unlabelled. waiting to be discovered.
"Starting from known initial variants, can the campaign reach the expensive ones?" rather than "can a fixed model
predict them?". `SCORE` is the axis that maps most directly onto what a screening campaign is for.

Parts:
* :class:`SplitStrategy`: turns a full pool into a :class:`SplitAssignment`. One per axis.
* :class:`ALSimulatorSplitDefinition`: strategy plus optional pool subsample.
* :class:`ALSimulatorSplit`: registry of named splits usable as a grid axis, mirroring
  :class:`ALSimulatorDataset`. ``FULL_POOL`` is the identity split and the default everywhere.

Strategies are pydantic models with a ``kind`` discriminator and are written to JSON with the rest of the campaign
config.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from collections import Counter
from enum import Enum, auto
from typing import (
    Annotated,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Union,
)

from pydantic import BaseModel, Field, model_validator
from biotrainer_core.data_classes import SequenceData


class ExtrapolationAxis(str, Enum):
    """The generalization question a split poses. See the table in the module docstring."""

    NUMBER = "number"
    """Train on variants with few substitutions, test on variants with more."""
    POSITION = "position"
    """Train on variants mutated at one set of positions, test on variants mutated elsewhere."""
    MUTATION = "mutation"
    """Train and test on disjoint sets of individual substitutions."""
    SCORE = "score"
    """Train on low-fitness variants, test on high-fitness ones."""
    WILDTYPE = "wildtype"
    """Train and test on variants of different parent sequences."""
    CUSTOM = "custom"
    """Composed or ad-hoc rule built from selectors."""


class ScoreEnd(str, Enum):
    """Which end of the label distribution the campaign starts from."""

    LOW = "low"
    """Train on the low end, test on the high end, for datasets where a larger label is better (MAXIMIZE)."""
    HIGH = "high"
    """Train on the high end, test on the low end. For MINIMIZE datasets, where the interesting
    variants are the small-label ones."""


class ReferenceStrategy(str, Enum):
    """How to obtain the parent sequence that substitution-based rules need."""

    EXPLICIT = "explicit"
    """Take it from the dataset definition. Raises if the dataset declares none."""
    CONSENSUS = "consensus"
    """Derive it as the per-position most common residue over the pool.

    Only valid for a **sparse** mutagenesis library, where most variants carry few substitutions so
    the parent residue is the clear majority at every position and can easily be recovered (e.g. in NucB epPCR
    library)."""


class SiteMatch(str, Enum):
    """How :class:`MutatedSites` compares a variant's mutated positions against the given ones."""

    ANY = "any"
    """At least one of the given positions is mutated."""
    ALL = "all"
    """Every given position is mutated; others may be too."""
    ONLY = "only"
    """The mutated positions are a subset of the given ones, nothing outside them changed."""


class SplitAssignment(BaseModel):
    """A partition of a pool into the campaign's starting set and the variants it must reach."""

    train_ids: List[str] = Field(
        min_length=1, description="Sequences the campaign starts from"
    )
    test_ids: List[str] = Field(
        min_length=1, description="Sequences withheld from the starting set"
    )
    description: str = Field(
        default="", description="Human-readable account of how this was derived"
    )

    @model_validator(mode="after")
    def _check_disjoint(self) -> SplitAssignment:
        overlap = set(self.train_ids) & set(self.test_ids)
        if overlap:
            raise ValueError(
                f"train and test overlap in {len(overlap)} sequences, e.g. {sorted(overlap)[:3]}"
            )
        return self

    def summary(self) -> str:
        total = len(self.train_ids) + len(self.test_ids)
        share = 100.0 * len(self.train_ids) / total
        return f"train={len(self.train_ids):,} ({share:.2f}%) test={len(self.test_ids):,} of {total:,}"


class SplitContext(BaseModel):
    """Everything a strategy may need beyond the sequences themselves."""

    reference_sequence: Optional[str] = Field(
        default=None,
        description="Parent sequence that substitutions are counted against",
    )
    wildtype_of: Optional[Dict[str, str]] = Field(
        default=None, description="seq_id -> parent identifier, for multi-WT datasets"
    )

    @staticmethod
    def build(
        pool: Sequence[SequenceData],
        explicit_reference: Optional[str] = None,
        strategy: ReferenceStrategy = ReferenceStrategy.EXPLICIT,
        wildtype_of: Optional[Dict[str, str]] = None,
    ) -> SplitContext:
        match strategy:
            case ReferenceStrategy.EXPLICIT:
                reference = explicit_reference
            case ReferenceStrategy.CONSENSUS:
                reference = consensus_sequence(pool)
            case _:
                raise ValueError(f"Unknown reference strategy: {strategy}")
        return SplitContext(reference_sequence=reference, wildtype_of=wildtype_of)

    def require_reference(self, kind: str) -> str:
        if self.reference_sequence is None:
            raise ValueError(
                f"Split '{kind}' is defined in terms of substitutions and therefore needs a parent sequence, but "
                f"the dataset does not declare one. Set `reference_fasta_path` on the dataset definition. Note that "
                f"the screening datasets are pools of unrelated sequences, where 'substitution' is undefined — only "
                f"the engineering (mutational landscape) datasets can use this axis."
            )
        return self.reference_sequence

    def require_wildtypes(self, kind: str) -> Dict[str, str]:
        if not self.wildtype_of:
            raise ValueError(
                f"Split '{kind}' partitions by parent sequence and therefore needs a seq_id -> wild-type mapping, "
                f"which this dataset does not provide. Only multi-wild-type datasets can use this axis (FLIP2's "
                f"`hydro` has three parents and `rhomax` several; every CombinGym landscape has exactly one)."
            )
        return self.wildtype_of


def consensus_sequence(
    pool: Sequence[SequenceData], min_modal_share: float = 0.5
) -> str:
    """Per-position most common residue over a fixed-length pool.

    Recovers the parent of a *sparse* mutational library.

    Notes:
    * mixed-length pools are refused.
    * every position's modal residue must hold at least `min_modal_share` of the pool. In a sparse
      library the parent residue dominates each position; in a combinatorially complete library it
      does not, and the mode is arbitrary.
    """
    lengths = {len(data_point.seq) for data_point in pool}
    if len(lengths) != 1:
        raise ValueError(
            f"Cannot derive a consensus parent: the pool has {len(lengths)} distinct sequence "
            f"lengths {sorted(lengths)[:5]}. A consensus is only meaningful for a fixed-length "
            f"mutational library, not for a pool of unrelated sequences."
        )

    consensus, weak = [], []
    for index, column in enumerate(zip(*(data_point.seq for data_point in pool))):
        counts = Counter(column)
        residue, hits = counts.most_common(1)[0]
        consensus.append(residue)
        share = hits / len(column)
        if share < min_modal_share:
            weak.append((index + 1, residue, share))

    if weak:
        shown = ", ".join(
            f"{position}:{residue}={share:.1%}" for position, residue, share in weak[:6]
        )
        raise ValueError(
            f"Refusing to derive a consensus parent: {len(weak)} position(s) have no dominant residue "
            f"({shown}{', ...' if len(weak) > 6 else ''}), so the mode is arbitrary there. This is what a "
            f"combinatorially complete designed library looks like — every residue is equally represented at "
            f"the varied sites, and the consensus would match the real parent at none of them. Declare the "
            f"parent explicitly via `reference_fasta_path` instead. (Lower min_modal_share only if you know "
            f"the library is sparse and merely noisy.)"
        )
    return "".join(consensus)


def substitutions(sequence: str, reference: str) -> FrozenSet[str]:
    """Substitutions of `sequence` against `reference`."""
    return frozenset(
        f"{expected}{index + 1}{actual}"
        for index, (expected, actual) in enumerate(zip(reference, sequence))
        if expected != actual
    )


def position_of(substitution: str) -> int:
    """1-based position of a substitution string."""
    return int(substitution[1:-1])


def _as_float(data_point: SequenceData) -> float:
    try:
        return float(data_point.label)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Sequence '{data_point.seq_id}' has a non-numeric label {data_point.label!r}, which the "
            f"score axis cannot order. Numeric labels are required for SCORE splits."
        ) from error


def _partition(
    pool: Sequence[SequenceData], in_train, description: str
) -> SplitAssignment:
    """Split `pool` by a per-sequence predicate, with a diagnostic error if either split is empty."""
    train_ids = [data_point.seq_id for data_point in pool if in_train(data_point)]
    test_ids = [data_point.seq_id for data_point in pool if not in_train(data_point)]
    if not train_ids or not test_ids:
        raise ValueError(
            f"Split '{description}' put all {len(pool):,} sequences on one side "
            f"(train={len(train_ids):,}, test={len(test_ids):,}). Both halves must be non-empty; "
            f"loosen the threshold or check that the dataset suits this axis."
        )
    return SplitAssignment(
        train_ids=train_ids, test_ids=test_ids, description=description
    )


class SequenceSelector(BaseModel):
    """A composable predicate over labelled sequences.

    Use cases:
    * **restricting the pool** by a criterion rather than at random (``pool_selector`` on a split
      definition), e.g. studying only the low-order corner of a landscape;
    * **combining conditions** via :class:`AllOf` / :class:`AnyOf` / :class:`Complement`, e.g. "at
      most two substitutions *and* above-median fitness";
    * **splitting on axes the benchmarks do not cover**, via :class:`SelectorSplit`, e.g. for discrete labels
      (:class:`LabelIn`).

    Subclasses implement :meth:`select`.
    """

    kind: str

    @abstractmethod
    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        """Return the subset of `pool` this selector admits, preserving input order."""

    def describe(self) -> str:
        fields = ", ".join(
            f"{name}={value!r}"
            for name, value in self.model_dump(exclude={"kind"}).items()
            if value is not None
        )
        return f"{self.kind}({fields})"


class MutationCount(SequenceSelector):
    """Variants carrying between `min_mutations` and `max_mutations` substitutions (inclusive).

    Unlike :class:`MutationNumberSplit` this is two-sided, so it can express a window like "variants
    with exactly two or three substitutions, excluding singles".
    """

    kind: Literal["mutation_count"] = "mutation_count"
    min_mutations: Optional[int] = Field(default=None, ge=0)
    max_mutations: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> MutationCount:
        if self.min_mutations is None and self.max_mutations is None:
            raise ValueError(
                "MutationCount needs at least one of min_mutations / max_mutations"
            )
        if (
            self.min_mutations is not None
            and self.max_mutations is not None
            and self.min_mutations > self.max_mutations
        ):
            raise ValueError("min_mutations must not exceed max_mutations")
        return self

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        reference = context.require_reference(self.kind)
        selected = []
        for data_point in pool:
            count = len(substitutions(data_point.seq, reference))
            if self.min_mutations is not None and count < self.min_mutations:
                continue
            if self.max_mutations is not None and count > self.max_mutations:
                continue
            selected.append(data_point)
        return selected


class LabelInterval(SequenceSelector):
    """Variants whose numeric label lies within [`lower_bound`, `upper_bound`] (inclusive).

    Two-sided, unlike :class:`ScoreSplit`'s single cut, so it can isolate a band of the fitness
    distribution instead of one tail.
    """

    kind: Literal["label_interval"] = "label_interval"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    @model_validator(mode="after")
    def _check_bounds(self) -> LabelInterval:
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError(
                "LabelInterval needs at least one of lower_bound / upper_bound"
            )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")
        return self

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        selected = []
        for data_point in pool:
            value = _as_float(data_point)
            if self.lower_bound is not None and value < self.lower_bound:
                continue
            if self.upper_bound is not None and value > self.upper_bound:
                continue
            selected.append(data_point)
        return selected


class LabelIn(SequenceSelector):
    """Variants whose (string) label is one of `labels`.

    The discrete counterpart of :class:`LabelInterval`, and the only way to build a split for a
    DISCRETE dataset.
    """

    kind: Literal["label_in"] = "label_in"
    labels: List[str] = Field(min_length=1)

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        wanted = set(self.labels)
        return [data_point for data_point in pool if str(data_point.label) in wanted]


class MutatedSites(SequenceSelector):
    """Variants whose mutated positions relate to `positions` as given by `match`."""

    kind: Literal["mutated_sites"] = "mutated_sites"
    positions: List[int] = Field(min_length=1)
    match: SiteMatch = SiteMatch.ANY

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        reference = context.require_reference(self.kind)
        wanted = set(self.positions)
        selected = []
        for data_point in pool:
            mutated = {position_of(s) for s in substitutions(data_point.seq, reference)}
            match self.match:
                case SiteMatch.ANY:
                    keep = bool(mutated & wanted)
                case SiteMatch.ALL:
                    keep = wanted <= mutated
                case SiteMatch.ONLY:
                    keep = mutated <= wanted
                case _:
                    raise ValueError(f"Unknown site match mode: {self.match}")
            if keep:
                selected.append(data_point)
        return selected


class SequenceLength(SequenceSelector):
    """Variants whose length lies within [`min_length`, `max_length`] (inclusive)."""

    kind: Literal["sequence_length"] = "sequence_length"
    min_length: Optional[int] = Field(default=None, ge=1)
    max_length: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> SequenceLength:
        if self.min_length is None and self.max_length is None:
            raise ValueError(
                "SequenceLength needs at least one of min_length / max_length"
            )
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("min_length must not exceed max_length")
        return self

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        selected = []
        for data_point in pool:
            length = len(data_point.seq)
            if self.min_length is not None and length < self.min_length:
                continue
            if self.max_length is not None and length > self.max_length:
                continue
            selected.append(data_point)
        return selected


class RandomSample(SequenceSelector):
    """A reproducible random subsample, as a composable selector.

    :class:`PoolSubsample` does the same thing as a plain pool pre-filter; this version can sit
    inside an :class:`AllOf` expression. Exactly one of `n` or `fraction`.

    Order-preserving; a pool smaller than `n` passes through unchanged.
    """

    kind: Literal["random_sample"] = "random_sample"
    n: Optional[int] = Field(default=None, ge=1)
    fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _check_size(self) -> RandomSample:
        if (self.n is None) == (self.fraction is None):
            raise ValueError("RandomSample needs exactly one of n / fraction")
        return self

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        size = (
            self.n if self.n is not None else max(1, round(len(pool) * self.fraction))
        )
        if size >= len(pool):
            return list(pool)
        chosen = set(random.Random(self.seed).sample(range(len(pool)), size))
        return [data_point for index, data_point in enumerate(pool) if index in chosen]


class AllOf(SequenceSelector):
    """Intersection: every child selector must admit the sequence."""

    kind: Literal["all_of"] = "all_of"
    selectors: List["AnySelector"] = Field(min_length=1)

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        selected = list(pool)
        for selector in self.selectors:
            selected = selector.select(selected, context)
        return selected

    def describe(self) -> str:
        return (
            f"all_of({', '.join(selector.describe() for selector in self.selectors)})"
        )


class AnyOf(SequenceSelector):
    """Unionr: at least one child selector must admit the sequence."""

    kind: Literal["any_of"] = "any_of"
    selectors: List["AnySelector"] = Field(min_length=1)

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        admitted: Set[str] = set()
        for selector in self.selectors:
            admitted.update(
                data_point.seq_id for data_point in selector.select(pool, context)
            )
        return [data_point for data_point in pool if data_point.seq_id in admitted]

    def describe(self) -> str:
        return (
            f"any_of({', '.join(selector.describe() for selector in self.selectors)})"
        )


class Complement(SequenceSelector):
    """Negation: everything the child selector rejects.

    A :class:`SplitStrategy` already yields both halves, so this is not needed to get a test set. It
    matters inside a composed expression: ``AllOf([MutationCount(max_mutations=2),
    Complement(LabelInterval(upper_bound=0.5))])``.
    """

    kind: Literal["complement"] = "complement"
    selector: "AnySelector"

    def select(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> List[SequenceData]:
        excluded = {
            data_point.seq_id for data_point in self.selector.select(pool, context)
        }
        return [data_point for data_point in pool if data_point.seq_id not in excluded]

    def describe(self) -> str:
        return f"complement({self.selector.describe()})"


AnySelector = Annotated[
    Union[
        MutationCount,
        LabelInterval,
        LabelIn,
        MutatedSites,
        SequenceLength,
        RandomSample,
        AllOf,
        AnyOf,
        Complement,
    ],
    Field(discriminator="kind"),
]

AllOf.model_rebuild()
AnyOf.model_rebuild()
Complement.model_rebuild()


class SplitStrategy(BaseModel):
    """Turns a pool into a :class:`SplitAssignment`. One subclass per extrapolation axis."""

    kind: str

    @property
    @abstractmethod
    def axis(self) -> ExtrapolationAxis:
        """Which generalization question this strategy poses."""

    @abstractmethod
    def assign(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> SplitAssignment:
        """Partition `pool` into a starting set and a withheld set."""


class MutationNumberSplit(SplitStrategy):
    """Train on variants with at most `max_train_mutations` substitutions, test on the rest."""

    kind: Literal["mutation_number"] = "mutation_number"
    max_train_mutations: int = Field(
        ge=0, description="Highest substitution count in the training half"
    )

    @property
    def axis(self) -> ExtrapolationAxis:
        return ExtrapolationAxis.NUMBER

    def assign(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> SplitAssignment:
        reference = context.require_reference(self.kind)
        limit = self.max_train_mutations
        return _partition(
            pool,
            lambda dp: len(substitutions(dp.seq, reference)) <= limit,
            f"{self.kind}: train on <={limit} substitutions",
        )


class PositionSplit(SplitStrategy):
    """Train on variants mutated only at some positions, test on variants touching the others.

    Give `test_positions` explicitly, `n_test_positions` to hold out that many of the dataset's mutated positions, or
    `test_fraction` to hold out that share of them.
    Held-out positions are chosen deterministically from `seed`.

    A variant joins the training half only if *every* position it mutates is a training position, so
    the halves are disjoint and exhaustive. The parent sequence mutates nothing and always trains.
    """

    kind: Literal["position"] = "position"
    test_positions: Optional[List[int]] = Field(
        default=None, description="1-based positions to hold out"
    )
    n_test_positions: Optional[int] = Field(
        default=None, ge=1, description="How many positions to hold out"
    )
    test_fraction: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Share of mutated positions to hold out",
    )
    seed: int = Field(default=42, description="Seed for choosing held-out positions")

    @model_validator(mode="after")
    def _check_choice(self) -> PositionSplit:
        given = [self.test_positions, self.n_test_positions, self.test_fraction]
        if sum(value is not None for value in given) != 1:
            raise ValueError(
                "PositionSplit needs exactly one of "
                "test_positions / n_test_positions / test_fraction"
            )
        return self

    @property
    def axis(self) -> ExtrapolationAxis:
        return ExtrapolationAxis.POSITION

    def assign(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> SplitAssignment:
        reference = context.require_reference(self.kind)
        per_variant = {
            dp.seq_id: {position_of(s) for s in substitutions(dp.seq, reference)}
            for dp in pool
        }

        if self.test_positions is not None:
            held_out = set(self.test_positions)
        else:
            mutated = sorted(
                {
                    position
                    for positions in per_variant.values()
                    for position in positions
                }
            )
            size = (
                self.n_test_positions
                if self.n_test_positions is not None
                else max(1, round(len(mutated) * self.test_fraction))
            )
            if size >= len(mutated):
                raise ValueError(
                    f"Cannot hold out {size} of only {len(mutated)} mutated positions "
                    f"({mutated}); at least one must remain for training."
                )
            held_out = set(random.Random(self.seed).sample(mutated, size))

        return _partition(
            pool,
            lambda dp: not (per_variant[dp.seq_id] & held_out),
            f"{self.kind}: hold out positions {sorted(held_out)}",
        )


class MutationSplit(SplitStrategy):
    """Train and test on disjoint sets of individual substitutions.

    Harder than :class:`PositionSplit`: the model sees a position during training but never the specific
    replacement it is asked about. Give `test_mutations` explicitly (``["A30S", ...]``), or
    `test_fraction` to hold out that share of the dataset's distinct substitutions, chosen
    deterministically from `seed`.

    A variant joins the training half only if *none* of its substitutions is held out.
    """

    kind: Literal["mutation"] = "mutation"
    test_mutations: Optional[List[str]] = Field(
        default=None, description="Substitutions to hold out, e.g. A30S"
    )
    test_fraction: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Share of distinct substitutions to hold out",
    )
    seed: int = Field(
        default=42, description="Seed for choosing held-out substitutions"
    )

    @model_validator(mode="after")
    def _check_choice(self) -> MutationSplit:
        if (self.test_mutations is None) == (self.test_fraction is None):
            raise ValueError(
                "MutationSplit needs exactly one of test_mutations / test_fraction"
            )
        return self

    @property
    def axis(self) -> ExtrapolationAxis:
        return ExtrapolationAxis.MUTATION

    def assign(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> SplitAssignment:
        reference = context.require_reference(self.kind)
        per_variant = {dp.seq_id: substitutions(dp.seq, reference) for dp in pool}

        if self.test_mutations is not None:
            held_out: Set[str] = set(self.test_mutations)
        else:
            distinct = sorted({s for subs in per_variant.values() for s in subs})
            size = max(1, round(len(distinct) * self.test_fraction))
            if size >= len(distinct):
                raise ValueError(
                    f"test_fraction={self.test_fraction} would hold out all {len(distinct)} distinct "
                    f"substitutions; at least one must remain for training."
                )
            held_out = set(random.Random(self.seed).sample(distinct, size))

        return _partition(
            pool,
            lambda dp: not (per_variant[dp.seq_id] & held_out),
            f"{self.kind}: hold out {len(held_out)} substitutions",
        )


class ScoreSplit(SplitStrategy):
    """Train on one end of the label distribution, test on the other.

    Start from mediocre measured variants and try to reach the good ones. Give `train_quantile`
    (0.8 -> the lowest-scoring 80% trains) or an absolute `train_threshold`.
    `train_on` flips the direction for MINIMIZE datasets.

    Ties at the boundary all land in the training half, so a heavily tied label distribution gives a
    larger training half than `train_quantile` suggests. The assignment description reports the
    realized sizes.
    """

    kind: Literal["score"] = "score"
    train_quantile: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Share of the pool, by rank, that trains",
    )
    train_threshold: Optional[float] = Field(
        default=None, description="Absolute label cutoff for training"
    )
    train_on: ScoreEnd = Field(
        default=ScoreEnd.LOW, description="Which end the campaign starts from"
    )

    @model_validator(mode="after")
    def _check_choice(self) -> ScoreSplit:
        if (self.train_quantile is None) == (self.train_threshold is None):
            raise ValueError(
                "ScoreSplit needs exactly one of train_quantile / train_threshold"
            )
        return self

    @property
    def axis(self) -> ExtrapolationAxis:
        return ExtrapolationAxis.SCORE

    def assign(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> SplitAssignment:
        scores = {dp.seq_id: _as_float(dp) for dp in pool}
        # Order so that "first" always means "trains": ascending for LOW, descending for HIGH.
        ascending = self.train_on is ScoreEnd.LOW

        if self.train_threshold is not None:
            threshold = self.train_threshold
        else:
            ordered = sorted(scores.values(), reverse=not ascending)
            index = min(
                len(ordered) - 1, max(0, round(len(ordered) * self.train_quantile) - 1)
            )
            threshold = ordered[index]

        def in_train(data_point: SequenceData) -> bool:
            score = scores[data_point.seq_id]
            return score <= threshold if ascending else score >= threshold

        end = "low" if ascending else "high"
        return _partition(
            pool,
            in_train,
            f"{self.kind}: train on the {end} end, threshold {threshold:g}",
        )


class WildTypeSplit(SplitStrategy):
    """Train on variants of some parent sequences, test on variants of another.

    The hardest axis and the one with the least data as it needs a dataset spanning several parents.
    Requires ``SplitContext.wildtype_of``.
    """

    kind: Literal["wildtype"] = "wildtype"
    test_wildtypes: List[str] = Field(
        min_length=1, description="Parent identifiers to hold out entirely"
    )

    @property
    def axis(self) -> ExtrapolationAxis:
        return ExtrapolationAxis.WILDTYPE

    def assign(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> SplitAssignment:
        wildtype_of = context.require_wildtypes(self.kind)
        held_out = set(self.test_wildtypes)
        known = set(wildtype_of.values())
        unknown = held_out - known
        if unknown:
            raise ValueError(
                f"Unknown wild type(s) {sorted(unknown)}; the dataset has {sorted(known)}."
            )
        if not known - held_out:
            raise ValueError(
                f"Holding out {sorted(held_out)} leaves no parent to train on."
            )

        return _partition(
            pool,
            lambda dp: wildtype_of.get(dp.seq_id) not in held_out,
            f"{self.kind}: hold out {sorted(held_out)}",
        )


class SelectorSplit(SplitStrategy):
    """Split on an arbitrary selector: whatever it admits trains, the rest is withheld.

    Covers conditions the benchmark axes do not like composed conditions or discrete labels.
    Reports :attr:`ExtrapolationAxis.CUSTOM`.
    """

    kind: Literal["selector"] = "selector"
    selector: AnySelector = Field(
        description="Sequences this admits become the starting set"
    )

    @property
    def axis(self) -> ExtrapolationAxis:
        return ExtrapolationAxis.CUSTOM

    def assign(
        self, pool: Sequence[SequenceData], context: SplitContext
    ) -> SplitAssignment:
        admitted = {
            data_point.seq_id for data_point in self.selector.select(pool, context)
        }
        return _partition(
            pool,
            lambda dp: dp.seq_id in admitted,
            f"{self.kind}: {self.selector.describe()}",
        )


AnySplitStrategy = Annotated[
    Union[
        MutationNumberSplit,
        PositionSplit,
        MutationSplit,
        ScoreSplit,
        WildTypeSplit,
        SelectorSplit,
    ],
    Field(discriminator="kind"),
]


class PoolSubsample(BaseModel):
    """A reproducible subsample applied *before* splitting.

    Sampling before the split keeps the train/test proportions of the split itself intact.

    Order-preserving; a pool already smaller than `n` passes through untouched.
    """

    n: Optional[int] = Field(default=None, ge=2)
    fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _check_size(self) -> PoolSubsample:
        if (self.n is None) == (self.fraction is None):
            raise ValueError("PoolSubsample needs exactly one of n / fraction")
        return self

    def apply(self, pool: Sequence[SequenceData]) -> List[SequenceData]:
        size = (
            self.n if self.n is not None else max(2, round(len(pool) * self.fraction))
        )
        if size >= len(pool):
            return list(pool)
        chosen = set(random.Random(self.seed).sample(range(len(pool)), size))
        return [data_point for index, data_point in enumerate(pool) if index in chosen]


class ALSimulatorSplitDefinition(BaseModel):
    """A named, reusable extrapolation split, optionally preceded by a pool subsample."""

    description: str = Field(description="What this split withholds, and why")
    strategy: Optional[AnySplitStrategy] = Field(
        default=None,
        description="None is the identity split: the whole pool is the starting set",
    )
    pool_selector: Optional[AnySelector] = Field(
        default=None,
        description="Restrict the pool by a criterion before splitting, e.g. to study one corner of "
        "a landscape in isolation. Applied before `subsample`.",
    )
    subsample: Optional[PoolSubsample] = Field(
        default=None,
        description="Shrink the pool before splitting, for landscapes too large to embed",
    )
    reference_strategy: ReferenceStrategy = Field(
        default=ReferenceStrategy.EXPLICIT,
        description="How to obtain the parent sequence that substitution-based rules need",
    )
    max_train_size: Optional[int] = Field(
        default=None,
        ge=2,
        description="Cap the starting set, moving the surplus into the withheld half",
    )
    train_sample_seed: int = Field(
        default=42, description="Seed for capping the starting set"
    )

    def is_identity(self) -> bool:
        return (
            self.strategy is None
            and self.subsample is None
            and self.pool_selector is None
        )

    @property
    def axis(self) -> Optional[ExtrapolationAxis]:
        return self.strategy.axis if self.strategy is not None else None

    def resolve(
        self,
        pool: Sequence[SequenceData],
        explicit_reference: Optional[str] = None,
        wildtype_of: Optional[Dict[str, str]] = None,
    ) -> tuple[List[SequenceData], Optional[SplitAssignment]]:
        """Return the (possibly restricted) pool and its train/test assignment.

        Order is: derive the parent sequence, narrow the pool by `pool_selector`, then `subsample`,
        then partition with `strategy`, then cap the starting set. The parent is derived from the
        *original* pool, so a consensus does not shift when the pool is narrowed.

        The identity split returns the pool unchanged and no assignment, so the campaign falls back
        to drawing a random starting set.
        """
        context = SplitContext.build(
            pool,
            explicit_reference=explicit_reference,
            strategy=self.reference_strategy,
            wildtype_of=wildtype_of,
        )

        resolved_pool = list(pool)
        if self.pool_selector is not None:
            resolved_pool = self.pool_selector.select(resolved_pool, context)
            if not resolved_pool:
                raise ValueError(
                    f"pool_selector {self.pool_selector.describe()} admitted 0 of "
                    f"{len(pool):,} sequences; nothing left to simulate on."
                )
        if self.subsample is not None:
            resolved_pool = self.subsample.apply(resolved_pool)

        if self.strategy is None:
            return resolved_pool, None
        assignment = self.strategy.assign(resolved_pool, context)
        return resolved_pool, self._cap_train(assignment)

    def _cap_train(self, assignment: SplitAssignment) -> SplitAssignment:
        """Shrink an oversized starting set, moving the surplus into the withheld half."""
        if (
            self.max_train_size is None
            or len(assignment.train_ids) <= self.max_train_size
        ):
            return assignment
        keep = set(
            random.Random(self.train_sample_seed).sample(
                assignment.train_ids, self.max_train_size
            )
        )
        surplus = [seq_id for seq_id in assignment.train_ids if seq_id not in keep]
        return SplitAssignment(
            train_ids=[seq_id for seq_id in assignment.train_ids if seq_id in keep],
            test_ids=assignment.test_ids + surplus,
            description=f"{assignment.description}, starting set capped at {self.max_train_size:,}",
        )


class ALSimulatorSplit(Enum):
    # The values are stored in the compressed dashboard data, so do not reorder these members
    FULL_POOL = auto()

    @staticmethod
    def all() -> List["ALSimulatorSplit"]:
        return list(ALSimulatorSplit)

    @staticmethod
    def for_axis(axis: ExtrapolationAxis) -> List["ALSimulatorSplit"]:
        return [split for split in ALSimulatorSplit if split.definition().axis is axis]

    def definition(self) -> ALSimulatorSplitDefinition:
        definition = _SPLIT_DEFINITIONS.get(self.name)
        if definition is None:
            raise ValueError(f"No split definition for {self.name}.")
        return definition

    def is_identity(self) -> bool:
        return self.definition().is_identity()

    def resolve(
        self,
        pool: Sequence[SequenceData],
        explicit_reference: Optional[str] = None,
        wildtype_of: Optional[Dict[str, str]] = None,
    ) -> tuple[List[SequenceData], Optional[SplitAssignment]]:
        return self.definition().resolve(
            pool, explicit_reference=explicit_reference, wildtype_of=wildtype_of
        )


_SPLIT_DEFINITIONS: Dict[str, ALSimulatorSplitDefinition] = {
    ALSimulatorSplit.FULL_POOL.name: ALSimulatorSplitDefinition(
        description="No restriction: the campaign may start anywhere in the pool.",
        strategy=None),
}
