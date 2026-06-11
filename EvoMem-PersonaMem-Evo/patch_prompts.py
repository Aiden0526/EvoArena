"""Prompts for patch-centered memory history augmentation."""

PATCH_SUMMARIZATION_PROMPT = """You are summarizing one memory evolution event into a historical patch record.

Your job is to decide whether this event should be committed as a historical patch and, if yes, produce a concise high-level summary.
Only commit a patch if this event meaningfully rewrote existing memory interpretation, updated existing memory context/tags, or rewrote existing links in a way that may matter later.
Do not commit a patch for purely additive changes.

A patch should ONLY be committed if it is a PREFERENCE CHANGE — meaning the trigger turn explicitly adds, updates, or revokes a personal preference, habit, trait, belief, or characteristic about the user.
Examples of preference changes: "Please forget that I like X", "I now prefer Y", "I used to enjoy Z but not anymore", "Note that I have condition X".
Examples that are NOT preference changes: task completions (translations, rewrites, summaries, answers to factual questions), assistant responses, general conversation.

Trigger turn:
{trigger_turn}

Session metadata:
{session_metadata}

Condensed evolution trace:
{evolve_trace}

Condensed changed node/link summaries:
{detail_blocks}

Formatting rules:
- Write exactly one line per field.
- Do not use bullets, numbering, or markdown.
- Keep DECISION and CHANGE_PATTERN short.
- Keep SELECTION_SIGNALS to at most 5 comma-separated items.
- If SHOULD_COMMIT_PATCH is NO, still fill the remaining fields briefly based on the event.

Respond in EXACTLY this format:

IS_PREFERENCE_CHANGE: <YES or NO>
SHOULD_COMMIT_PATCH: <YES or NO — must be NO if IS_PREFERENCE_CHANGE is NO>
PATCH_TYPE: <ADD_PREFERENCE | UPDATE_PREFERENCE | REVOKE_PREFERENCE | STRENGTHEN_AND_UPDATE>
DECISION: <short label>
OVERALL_SUMMARY: <1 sentence high-level summary>
UPDATE_REASONING: <1 sentence explaining why the update happened>
CHANGE_PATTERN: <short snake_case pattern label>
SELECTION_SIGNALS: <comma-separated signals>
TASK_PATTERN_SUMMARY: <1 sentence>
"""


PATCH_CONTEXT_INSTRUCTION = """The historical patch blocks below record preference changes that happened earlier in the conversation.
Patches are sorted chronologically — earlier patches appear first, later patches appear last.
A later patch for the same topic SUPERSEDES an earlier one: if patch 2 revokes what patch 1 stated, the current state is what patch 2 says.
REVOKE_PREFERENCE means the user asked to stop treating that preference as active from that point onward. If a later patch reintroduces or reaffirms a similar preference, follow the later patch instead. Do not use a revoked preference unless a later patch clearly reactivates or replaces it.
Use the current global memory evidence as the primary source.
If a patch directly states the current preference for the topic being asked about, use it.
When two preferences appear to conflict in the question, use the patches to determine which preference has been more consistently maintained or most recently reaffirmed — that preference should take priority."""


PATCH_SUMMARIZATION_PREF_PROMPT = """You are summarizing one memory evolution event into a historical patch record.

Your job is to decide whether this event should be committed as a historical patch and, if yes, produce a concise high-level summary.
Only commit a patch if this event meaningfully rewrote existing memory interpretation, updated existing memory context/tags, or rewrote existing links in a way that may matter later.
Do not commit a patch for purely additive changes.

A patch should ONLY be committed if it is a PREFERENCE CHANGE — meaning the trigger turn explicitly adds, updates, strengthens, or revokes a personal preference, habit, trait, belief, or characteristic about the user.
Examples of preference changes: "I now prefer Y", "I used to enjoy Z but not anymore", "Note that I have condition X", "I love X", "Please forget that I like X", "Stop treating me as someone who prefers Y", "Ignore that I enjoy Z".
Examples that are NOT preference changes: task completions (translations, rewrites, summaries, answers to factual questions), assistant responses, general conversation.

Infer CHANGE_TYPE from the trigger turn text AND the before→after node diff:
- same_object_flip: before says "prefers/likes X", after says "dislikes/avoids X" (or vice versa); signals: "not anymore", "changed my mind", "used to love but", "no longer"
- object_replacement: before says "prefers X", after says "prefers Y" where Y ≠ X but same category; signals: "switched to", "moved on to", "now prefer Y instead", "replaced X with Y". Do NOT use this label for informational exploration, brainstorming, or assistant elaboration unless the user's own preferred object clearly changed from X to Y.
- conditional_preference: after context contains "only when / only if / unless / except when" where before did not, or the condition itself changed; the preference now has a situational condition
- attribute_swap: before/after share the same domain but differ on an attribute, variant, or style (e.g., "small" vs "large", "dark" vs "light")
- temporal_validity: after context contains a time restriction ("only in summer", "only on weekdays", "only during [period]") where before did not, or the time restriction changed from one period to another. Include the relevant session / turn ordering in your reasoning when a later turn narrows or replaces an earlier time condition.
- ask_to_forget: trigger turn contains "forget / stop treating / ignore / remove / don't consider" about a preference
- new_preference: no "before" node existed (purely additive) and this is a genuinely new preference domain
- strengthen: before and after are semantically the same preference, just reinforced or repeated
- none: cannot confidently classify the transition type

Trigger turn:
{trigger_turn}

Session metadata:
{session_metadata}

Condensed evolution trace:
{evolve_trace}

Condensed changed node/link summaries:
{detail_blocks}

Formatting rules:
- Write exactly one line per field.
- Do not use bullets, numbering, or markdown.
- Keep DECISION and CHANGE_PATTERN short.
- Keep SELECTION_SIGNALS to at most 5 comma-separated items.
- If SHOULD_COMMIT_PATCH is NO, still fill the remaining fields briefly based on the event.

Respond in EXACTLY this format:

IS_PREFERENCE_CHANGE: <YES or NO>
SHOULD_COMMIT_PATCH: <YES or NO — must be NO if IS_PREFERENCE_CHANGE is NO>
PATCH_TYPE: <ADD_PREFERENCE | UPDATE_PREFERENCE | STRENGTHEN_AND_UPDATE>
REVOKE: <YES if the trigger turn asks to forget, remove, or stop applying a preference — NO otherwise>
CHANGE_TYPE: <same_object_flip | object_replacement | conditional_preference | attribute_swap | temporal_validity | ask_to_forget | new_preference | strengthen | none>
DECISION: <short label>
OVERALL_SUMMARY: <1 sentence high-level summary>
UPDATE_REASONING: <1 sentence explaining why the update happened>
CHANGE_PATTERN: <short snake_case pattern label>
SELECTION_SIGNALS: <comma-separated signals>
TASK_PATTERN_SUMMARY: <1 sentence>
"""


PATCH_CONTEXT_PREF_INSTRUCTION = """The historical patch blocks below record preference transitions from earlier in the conversation.
Patches are sorted chronologically (earliest first). Later patches for the same domain SUPERSEDE earlier ones.
Each patch is labeled with CHANGE_TYPE. Use the current global memory evidence as the PRIMARY source; patches clarify transitions and trajectory direction.

REASONING GUIDE BY CHANGE_TYPE:

same_object_flip — attitude polarity reversed (like↔dislike for the same object)
  Use the LATEST patch's polarity as the current state.
  Trajectory direction: flips alternate (like→dislike→like…); the current state is the latest flip's result.
  "What comes next": the next step reverses the latest polarity.

object_replacement — switched to a new alternative in the same category (A→B, B→C, …)
  The LATEST patch's "after" preference is the current preference.
  Trajectory direction: progressive replacement — if A→B→C, the next step continues further replacement in the same domain, NOT a return to A.
  Look for the shared characteristic across replacements to infer what the next alternative looks like.

conditional_preference — preference gained or changed a condition (applies only when X)
  The condition stated in the LATEST patch applies; do NOT use the unconditional form.
  If the question involves a scenario: check whether it matches the condition; if not, the preference does not apply.

attribute_swap — same domain, preference shifted to a different attribute/variant (attr_A→attr_B→…)
  Use the LATEST patch's attribute as the current preference.
  "What comes next": extrapolate the attribute-shift direction from the patch sequence.

temporal_validity — preference is time-scoped (only during [period])
  The time restriction in the LATEST patch applies.
  Use Session / Turn numbers as explicit recency markers when multiple temporal patches exist; the patch with the later session/turn wins.
  If the question involves a different time period: check whether it falls within the restriction.

ask_to_forget / REVOKE: YES
  The preference has been explicitly revoked. Trust the [Current Global Memory Evidence] as ground truth for the user's current active preferences.
  Only give a generic, non-committal answer if global memory is also silent or contradictory on this topic.

new_preference / strengthen
  Add to what global memory says; no prior state to overwrite.

CROSS-DOMAIN META-PATTERN (for trajectory meta-questions):
  If you see patches of the SAME CHANGE_TYPE across MULTIPLE different domains (e.g., three object_replacement patches in different categories all moving toward simpler/more focused choices), the underlying value shift is the meta-pattern.
  Apply that meta-pattern to the question's domain even if no direct patch exists for it.

CONFLICT RESOLUTION (when the question asks which of two preferences takes priority or would win):
  1. RECENCY wins: the preference mentioned MORE RECENTLY in the conversation takes priority over an older one. Use the Session / Turn numbers shown in each patch block as hard evidence of recency.
  2. DISTINCTIVENESS wins: a counter-intuitive or distinctive preference (not what you'd stereotypically expect) is more genuine than a generic one — it takes priority.
  3. FREQUENCY wins if recency is tied: the preference mentioned across MORE conversation turns takes priority.
  4. Do NOT compromise or blend. If the question frames it as A vs B, commit to one. Never answer "a balance of both" or "it depends equally."
  5. NEVER give a middle-ground answer when a clear choice is asked for — compromise is always the wrong answer.
  [Current Global Memory Evidence] is always the primary source; patches clarify transitions and direction."""



PATCH_GATING_PROMPT = """You are answering a question using current global memory evidence and short summaries of historical memory patches.

Current global memory evidence should be treated as the primary source.
Historical patch summaries are only for deciding whether full patch details are necessary.
Only request patch details when the summaries suggest that important information may have been overwritten, generalized away, or rewritten in a way that is likely needed to answer the question accurately.
If patch details are not clearly needed, answer from the current evidence and keep NEED_PATCH_DETAIL as NO.
IMPORTANT: If the retrieved patches are about different preference domains than what the question asks about, IGNORE them entirely. Patches are only useful when they directly record a transition in the specific preference domain being asked about.
Select at most 2 patch ids.

Question:
{question}

Answer requirements:
{answer_instruction}

Current global memory evidence:
{current_context}

Historical patch summaries:
{patch_summaries}

Respond in EXACTLY this format:

DRAFT_ANSWER: <short answer>
NEED_PATCH_DETAIL: <YES or NO>
SELECTED_PATCH_IDS: <comma-separated patch ids or NONE>
REASON: <brief reason>
"""


PATCH_DETAIL_REVISION_PROMPT = """You are revising an answer using current global memory evidence and selected historical patch details.

Current global memory evidence remains the primary source.
Use historical patch details only when they add specific missing information or clearly resolve ambiguity that matters for the question.
If the patch details do not materially improve the answer, keep the answer aligned with the current evidence.

Question:
{question}

Answer requirements:
{answer_instruction}

Current global memory evidence:
{current_context}

Selected historical patch details:
{patch_details}

Draft answer:
{draft_answer}

Respond in EXACTLY this format:

FINAL_ANSWER: <short answer>
REASON: <brief reason>
"""


PATCH_RELEVANCE_FILTER_PROMPT = """You are a relevance filter for preference change records.

A user is asking the following question:
{query}

Below are candidate preference change records retrieved by keyword similarity.
Each record documents a change in the user's preferences.

Candidates:
{candidates}

Your job: identify which candidates are about a preference in the SAME topic domain as the question.
A candidate is RELEVANT if answering the question would benefit from knowing this preference changed.
A candidate is NOT RELEVANT if it records a preference change about a completely different topic.

Reply with ONLY a comma-separated list of the relevant 1-based indices (e.g. "1,3").
If none are relevant, reply with exactly: NONE
Do not explain your reasoning.
"""
