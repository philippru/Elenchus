% ============================================================================
% Elenchus — SCC symbolic verification layer (Clingo / ASP)
% ----------------------------------------------------------------------------
% This file holds ONLY the static rule set. Per-example facts (is_a/2,
% has_property/2, antonym/2, part_of/2, causes/2, obstructs/2, label/1) are
% generated dynamically by pipeline.py and prepended to this program before
% grounding. Keep the rules minimal and extensible — document every rule.
% ============================================================================

% ---------------------------------------------------------------------------
% ENTAILMENT SUPPORT
% Two distinct impactful arguments that share a category or a property are
% taken as evidence that the hypothesis is *entailed* by the premise (they
% denote semantically compatible things).
% ---------------------------------------------------------------------------
supports_entailment :- is_a(X, Y), is_a(Z, Y), X != Z.            % shared category
supports_entailment :- has_property(X, P), has_property(Y, P), X != Y.  % shared property
supports_entailment :- part_of(X, W), part_of(Y, W), X != Y.      % shared whole (mereology)

% ---------------------------------------------------------------------------
% CONTRADICTION SUPPORT
% An antonym relation among the impactful arguments — directly, or via two
% properties that are antonyms of each other — is evidence of contradiction.
% A `causes`/`obstructs` clash is treated as a (weaker) contradiction signal.
% ---------------------------------------------------------------------------
supports_contradiction :- antonym(T1, T2).                        % direct antonymy
supports_contradiction :- antonym(X, Y),                          % property-level antonymy
                          has_property(T1, X),
                          has_property(T2, Y).
supports_contradiction :- causes(X, E), obstructs(Y, E), X != Y.  % one causes what another blocks

% ---------------------------------------------------------------------------
% NEUTRAL SUPPORT
% Neutral is the residual: no entailment and no contradiction evidence found.
% (Default negation under the stable-model semantics; the program is
% stratified, so the answer set is unique.)
% ---------------------------------------------------------------------------
supports_neutral :- not supports_entailment, not supports_contradiction.

% ---------------------------------------------------------------------------
% CONSISTENCY CHECK
% SCC is "consistent" iff the symbolic support matches the gold/predicted
% label that was asserted as label/1.
% ---------------------------------------------------------------------------
scc_consistent :- label(entailment),    supports_entailment.
scc_consistent :- label(contradiction), supports_contradiction.
scc_consistent :- label(neutral),       supports_neutral.
scc_inconsistent :- not scc_consistent.

% Expose only the verdict + the support atoms (keeps the model readable).
#show scc_consistent/0.
#show scc_inconsistent/0.
#show supports_entailment/0.
#show supports_contradiction/0.
#show supports_neutral/0.
