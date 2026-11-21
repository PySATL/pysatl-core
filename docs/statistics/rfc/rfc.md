# Sample Statistics Architecture in PySATL Core

Concise architecture proposal · 19 November 2026

**Status: Draft, proposal for discussion.** The APIs and behavior described here define requirements for the system under development.

## Terminology

- **Sample** - a protected snapshot of observations that preserves their order and duplicates.
- **Statistic** - an immutable configuration of an operation on a sample, such as variance with a specified `ddof`.
- **Session (SampleSession)** - an object for repeated and composite computations on a single snapshot.
- **Dependency** - another statistic requested by a statistic through the session's internal `get_statistic` method during the current invocation.
- **Computed** - a numerical result with its mathematical warnings and provenance.
- **StatisticResult** - a public result with its configuration and diagnostics for the current invocation.
- **Statistic key (`Statistic.key: str`)** - a string containing the statistic type and every setting that affects applicability validation or the computed result. For example, the keys for `Variance(ddof=0)` and `Variance(ddof=1)` must differ.
- **Cache entry identifier** - the composite pair `(type(statistic), statistic.key)` used by a session to distinguish entries. This is a separate concept; only `statistic.key` is the statistic's string key.

## Research summary

`Sample.prepare` validates the input and creates a protected sample snapshot. The applicability of a particular statistic to that sample is checked separately. A shared dependency mechanism allows expensive operations, such as sorting, to be reused and applies the same validation rules to built-in and third-party statistics.

## Considered options

| Option | Advantage | Limitation |
|---|---|---|
| Only functions over arrays | Simple one-off calls | Shared dependencies, caching, and diagnostics would need to be managed separately |
| Statistics and cache inside Sample | A concise entry point | The data snapshot acquires computation state and responsibility for the cache lifetime. Sample must remain lightweight; storing statistics and a cache conflicts with this requirement |
| Immutable Sample, Statistic, and a separate session | Independent data and configuration, explicit cache ownership | Requires a shared execution contract |

The third option is selected: one snapshot can be used in different sessions, and one configuration can be applied to different data. Numerical functions remain a lower layer and can be called without result objects. For ordinary use, `statistic.compute(sample)` is sufficient.

## Purpose and scope

The module must compute characteristics of observations without specifying a probability family: the mean, variance, moments, quantiles, order statistics, and L- and TL-statistics. A sample mean and a model's expected value have different inputs and existence conditions, so statistics form a separate module alongside estimation and distributions.

The first version operates on one non-empty sample of finite real-valued observations. Streaming, observation weights, multidimensional input, and concurrent public invocations within one session are out of scope. The i.i.d. assumption is not checked automatically.

## Architecture

```mermaid
classDiagram
    direction TB

    class Sample {
        +prepare(data) Sample
        +values
        +n
    }
    class SampleSession {
        +sample
        +values
        +n
        +cache_enabled: bool
        +compute(statistic, cache=True) StatisticResult
        ~get_statistic(statistic) Computed
        ~add_warning(warning)
        +clear_cache()
    }
    class Statistic {
        <<abstract>>
        +name
        +key: str
        +compute(sample) StatisticResult
        #_validate(session)
        #_calculate(session) Computed
    }
    class Variance {
        +ddof
    }

    SampleSession --> "1" Sample : uses snapshot
    SampleSession ..> Statistic : calls hooks
    Statistic <|-- Variance
```

- **Sample** owns a protected snapshot of observations. It does not create a session or store statistics or a cache. One snapshot can support independent computations.
- **Statistic[V]** stores an immutable configuration and provides the shared `compute(sample)` method, which always runs without caching. Subclasses define `name`, a string `key` containing the type and all relevant settings, `_validate(session) -> None`, and `_calculate(session) -> Computed[V]`. In both hooks, all requests for other statistics go through `session.get_statistic(...)` within the current invocation and under its cache policy. The shared execution order must not be overridden; statistics do not require registration.
- **SampleSession** manages execution, dependencies, caching, and diagnostics for one Sample. To use caching, the user explicitly creates a session and calls `session.compute(statistic)`. Statistics use the internal `get_statistic` method to obtain dependencies; the session lifetime determines how long entries are retained and when they are released.
- **Numerical kernels** operate on arrays and numbers without sessions or public result objects. Estimators can use them without constructing a `StatisticResult`.

In UML, `~` on `get_statistic` denotes an internal module operation. This method is not a user entry point.

A session accepts a prepared Sample and retains it without copying for its entire lifetime; the snapshot cannot be replaced. `session.sample`, `session.values`, and `session.n` provide statistics with read-only access. The permanent policy is set through `SampleSession(sample, cache=True)` and exposed as immutable `cache_enabled`; changing it after construction is not supported. Memory policy and the depth limit belong to the session, rather than Sample or Statistic.

## Data rules and public API

Both public methods, `statistic.compute(sample)` and `session.compute(statistic)`, return `StatisticResult[V]`. The internal `session.get_statistic(statistic)` method returns `Computed[V]`.

`Sample.prepare(data)` creates an independent snapshot of a one-dimensional `float64` sample, preserving order and duplicates. Strings, bool, complex, object arrays, masked arrays, empty input, NaN, and infinities are rejected. `sample.values` has shape `(n,)`; `sample.n` is the observation count.

Changing the original array or accessing `values` must not mutate the snapshot; otherwise, cached entries would no longer correspond to the observations. `Sample.prepare(sample)` returns the same object; `sample[1:4]` and even `sample[:]` create a new protected Sample. Empty slices are rejected under the general rules.

An ordinary `statistic.compute(sample) -> StatisticResult` call always runs without caching, including validation and all dependencies. Caching cannot be enabled through `Statistic.compute`. Dependencies use a temporary internal session with caching disabled and an explicit invocation-level `cache=False`: no entry storage is allocated, read, or written. The internal session is released after the call; neither Sample nor Statistic retains it. Repeated ordinary calls do not share entries. Its result's `CacheInfo` has `requested=False`, `effective=False`, and `from_cache=False`; no `session_cache_disabled` warning is produced.

To use caching, the user explicitly creates `session = SampleSession(sample)` and calls `session.compute(statistic, cache=True)`. Passing `cache=False` disables reads and writes for the entire dependency chain. A statistic's configuration does not retain a sample or invocation state.

```python
# Proposed API.
sample = Sample.prepare([1, 2, 3])
direct = Variance(ddof=1).compute(sample)  # Always without caching.
session = SampleSession(sample)
first = session.compute(Variance(ddof=1))
second = session.compute(Variance(ddof=1))
fresh = session.compute(Variance(ddof=1), cache=False)
```

## Interaction sequence

Scenario: an explicitly created SampleSession with caching enabled computes a central moment with a Mean dependency. An ordinary `statistic.compute(sample)` call has caching disabled. A ready root result bypasses hooks and dependency requests.

```mermaid
sequenceDiagram
    participant U as User
    participant S as SampleSession
    participant C as CentralMoment
    participant M as Mean

    U->>S: compute(CentralMoment(3))
    S->>C: _validate(session)
    S->>C: _calculate(session)
    C->>S: get_statistic(Mean()) - internal call
    alt Mean is available in the cache
        S-->>C: Cached Computed
    else Mean must be computed
        S->>M: _validate if needed<br/>Then _calculate(session)
        M-->>S: Computed
        S-->>C: Computed
    end
    C-->>S: Computed central moment
    S-->>U: StatisticResult
```

### Validation and dependencies

**Data preparation → invocation start → applicability validation → calculation → public result → invocation-state cleanup.** A statistic's constructor validates settings, Sample validates data, `_validate` checks applicability to the sample, and `_calculate` and `Computed` check the numerical result. A validation failure prevents the main calculation; no prepared object is passed between the stages.

Both stages request dependencies through `session.get_statistic(statistic) -> Computed`. This is an internal method for interaction between statistics and the session: `_validate` and `_calculate` implementations call it, not users. The user entry points are `statistic.compute(sample)` and `session.compute(statistic)`. The `get_statistic` method is available only within an active invocation and inherits its cache policy, stack, and diagnostics. Calling another statistic's public `compute` would start an independent invocation and lose the shared stack. Without caching, a dependency needed by both stages may be computed more than once.

Outside an active invocation, `get_statistic` and `add_warning` are rejected. `_validate` returns only `None`; it does not store prepared data in Statistic. `_calculate` requests its own dependencies and returns `Computed`, rather than a public result. Statistic authors do not read or modify cache entries. `session.add_warning(...)` adds a mathematical warning to the current request; the executor merges warnings from validation, calculation, and dependencies used.

A cycle is detected when an active identifier `(type(statistic), statistic.key)` repeats, for example `A → B → A`, **before reading the cache**. The identifier remains active from the start of validation until calculation finishes; a set detects repetition, a stack explains the path, and `finally` removes active markers on every exit. Requesting a dependency that has already completed is allowed. A completed-result cache or LRU does not replace this protection: an unfinished calculation has no result yet.

An active-depth limit additionally bounds chains of distinct identifiers; an earlier Python `RecursionError` remains a separate failure. Overlapping public invocations, re-entry, and cache clearing during an active session invocation are rejected immediately. Diagnostics for a result or an expected error are copied before working state is released.

The limit counts simultaneously active requests, including the root, and is checked before adding the next request to the active stack. A detected cycle raises `StatisticDependencyCycleError` with the dependency path; exceeding the session's own limit raises `StatisticDependencyLimitError`. `RecursionError` is not relabeled as either. An internal request removes its own active marker and stack frame in `finally`; the public boundary releases all invocation working state, including on unexpected exceptions. The same session can accept the next public call; successful dependency entries are not rolled back when their parent fails.

## Cache entry identifier, states, and policy

A session uses the entry identifier `(type(statistic), statistic.key)`: the first element is the statistic's class object, and the second is its string key. `Statistic.key` contains the type designation and every setting that affects validation or the result: `ddof`, orders, levels, trimming, conventions, and algorithm settings. Identical configurations of the same type must produce identical strings; configurations that differ in relevant settings must produce different strings. The sample is not part of the entry identifier because the session is already bound to a single immutable snapshot. Different sessions do not share entries.

The key is immutable throughout the Statistic's lifetime and does not depend on object addresses, randomized hashes, or call history. Requested component order and duplicates are significant; mutable settings, including weight arrays, are copied and protected when the configuration is constructed. Within one session, the result type and shape are fixed for each type/key pair. The string encoding is an implementation detail, but it must unambiguously distinguish relevant configurations.

- **No entry:** perform validation and calculation.
- **Valid without a result:** validation was retained, but the subsequent calculation failed or the result was not stored because of the memory budget. Reuse validation and perform the calculation.
- **Valid with Computed:** return the ready result without repeating dependency requests.
- **Invalid without a result:** create a fresh exception from the retained reason; do not run the calculation. Invalid combined with a result is forbidden.

A separate validation outcome is needed because validation itself may require expensive dependencies. Invalid retains only expected, deterministic applicability failures, without retaining an exception object or traceback. Numerical failures, programming errors, cycles, and depth-limit failures are excluded, including when they originate in a dependency during validation.

Classification considers the original failure phase and reason, rather than just the caught exception's class: a dependency's calculation error does not become its parent's applicability failure merely because the parent requested it in `_validate`.

State transitions are defined as follows:

| Event | Outcome and storage when caching is enabled |
|---|---|
| `_validate` returned `None` | Construct Valid with validation warnings and attempt to store it before calculation. Valid contains no prepared data. |
| Expected deterministic applicability failure | Construct Invalid with an immutable description of the known error kind, reason, and warnings; do not calculate. A dependency failure retains its original error kind and phase. |
| Unexpected or numerical validation failure | Do not create Invalid for the parent; propagate the error while ensuring working-state cleanup. |
| `_calculate` returned a valid Computed | Atomically store or replace the whole entry with Valid plus the result. |
| Calculation failure or invalid Computed | Do not store a result; an already stored Valid and successful dependencies may remain. |

Failure to store validation does not prevent calculation; failure to store Invalid does not prevent raising the error. A stored result must include Valid in the same entry, even if the separate attempt to store validation failed. An entry constructor or internal factory enforces the invariant that a result requires Valid. Reusing Invalid creates a new typed exception; arbitrary classes are not imported from a retained string.

`effective = session.cache_enabled AND invocation_cache` applies to the entire chain. False disables reads and writes, including validation outcomes, but preserves existing entries. True when caching is permanently disabled produces a `session_cache_disabled` warning; an explicit False does not. `get_statistic` has no separate cache argument.

Storage is allocated lazily. The budget accounts for all retained entry identifiers, validation outcomes, values, metadata, and buffers; failing to store an entry does not change the result. Eviction and `clear_cache()` remove whole entries while preserving Sample and the policy. The cache budget does not limit process memory: the snapshot, temporary arrays, and results held by the caller are outside it.

This also applies to auxiliary caches for sorting, powers, and coefficients: hidden global storage must not bypass `cache=False`. Temporary computations within the current request are allowed without retaining completed dependencies for reuse.

Storing or atomically replacing an entry must not exceed the specified budget. Accounting includes warnings, details, and Invalid descriptions; even a small validation entry may not fit. Conservative accounting of shared buffers is allowed, but a retained buffer must not stop being counted after another entry is evicted. If an expanded result entry does not fit, the previous Valid without a result may remain; diagnostics separately record the storage refusal as `memory_limit`. Specific budget values, storage layout, and eviction order are implementation details; this rule does not conceal arbitrary memory-allocation failures during calculation itself.

`clear_cache()` idempotently releases every entry reference and resets memory accounting, returning storage to its unallocated state; clearing an empty cache does not allocate a container. Active markers are not cache entries and cannot be evicted. Previously returned results survive clearing; their buffers are released only after all retaining references disappear.

## Results and diagnostics

```mermaid
classDiagram
    direction TB

    class Computed {
        +value
        +components
        +n_used
        +warnings
    }
    class StatisticResult {
        +value
        +statistic
        +n_observations
        +warnings
    }
    class StatisticCacheEntry {
        +validation
        +result
    }
    class Validation {
        Valid or Invalid
    }
    class ComputationDetails {
        +algorithm
        +route
        +dtype
    }
    class CacheInfo {
        +requested
        +effective
        +from_cache
        +entries
    }

    StatisticCacheEntry --> "1" Validation : validation outcome
    StatisticCacheEntry --> "0..1" Computed : ready result
    Computed --> ComputationDetails : numerical provenance
    StatisticResult --> ComputationDetails : details
    StatisticResult --> CacheInfo : cache for current invocation
```

Computed and StatisticResult contain a value and its description; the public result is assembled from Computed but is not its subclass. An Invalid entry has no result value. The diagram shows relationships between objects rather than the call sequence.

`Computed` contains the value, components, the number of observations used (`n_used`), numerical `details`, and mathematical warnings. The public `StatisticResult` preserves these fields and adds `statistic` (the current request's immutable configuration), `n_observations` (the original sample size), a fresh `cache: CacheInfo`, and warnings from the current invocation. The retained numerical result is separate from the report of how it was used; even a ready root cache hit creates a new public report.

A scalar request gives a numerical `value` within `StatisticResult`, without changing the public result type. A vector `value` is a `float64` array of shape `(k,)`, preserving order and duplicates even for `k=1`. `components` is an immutable sequence of labels in the same order: one for a scalar, `k` for a vector. Empty requests are rejected. Computed and StatisticResult constructors check finiteness and agreement between shape and components; invalid output cannot enter the cache or a successful public result. A component failure produces an error identifying its position, without returning a partial vector.

`n_used` records the number of observations actually used: trimming may reduce it, but zero rank weights alone do not. Configuration, values, components, details, warnings, and CacheInfo are immutable, including nested objects. A frozen dataclass or `writeable=False` on an accessible mutable buffer alone is insufficient: changing an input array, view, or accessible backing memory must not change Sample, Computed, a returned result, or a cache entry. Defensive copies or sharing genuinely immutable memory are allowed. Diagnostics do not retain Sample, the session, working lists, or internal cache entries.

`details` explains the actual algorithm and the `direct` / `fallback` route; it does not form a recursive tree of dependency details. Mathematical warnings are retained with validation or a value and remain available on reuse. Policy warnings belong only to the current invocation. Previously returned results and reports do not change after subsequent computations or cache clearing.

Computed also includes mathematical warnings from dependencies used only in `_validate`; a ready root retains them without rerunning those dependencies. Transferring the same warning repeatedly does not duplicate it, but matching text does not merge different sources or components; ordering is reproducible. Before cleanup, an expected error receives independent `error.warnings` and `error.cache`; an error produces no successful StatisticResult. Policy warnings and old invocation events are not stored in Computed, Valid, or Invalid.

`from_cache` means a ready root result was used, `stored` means the numerical result was stored in the current invocation, and `validation_stored` means validation was stored. Using a cached mean in a new variance calculation does not make that variance `from_cache=True`. A ready root result does not generate fictitious dependency events.

`CacheInfo` records the permanent `session_enabled` policy, `requested`, the `effective` mode, root `from_cache`, and immutable `entries` for requests actually made. Each request event separately records the result source, validation source/outcome, `stored`, and `validation_stored`. A Valid-only hit is not a ready result; with ready Computed, validation was not executed in this request. These flags describe current-request actions, rather than entry presence after later evictions.

**Checks before acceptance:** independence of snapshots and sessions; equal values with caching enabled and disabled; completeness and reproducibility of the string `Statistic.key`; type distinction in the entry identifier; all cache states; cycles in both stages with caching disabled; cleanup after errors; vector shape and warning propagation; numerical stability on boundary inputs; memory accounting and fresh reports. Agreement with a single third-party library is insufficient.
