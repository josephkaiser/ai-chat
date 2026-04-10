# Feedback-Driven Improvement Case Study

Date: 2026-04-10

Baseline commit while writing: `020e515` (`updates to app.py`)

Related recent commits in the same iteration cycle:

- `77788e5` (`updates to app.py and permissions`)
- `acb594c` (`docs update`)

Note: the line references below point at the current working tree on top of `020e515`. That is the most useful way to review the code right now, even though some of the latest changes are still uncommitted.

## Representative Prompt

From `data/chat.db`, conversation `343c7b07-ab29-4e2d-a707-4d94795d274a`:

> Build a small Python script in this repo workspace that generates a matplotlib PNG from sample data, run it yourself, and show me the actual image artifact in the viewer.

Initial assistant output:

> Here is the generated matplotlib plot from the sample data:
>
> `[[artifact:Matplotlib_Plot.html]]`

This looked superficially correct, but it exposed a product failure:

1. The artifact reference appeared in chat as text instead of feeling like a first-class UI action.
2. The viewer could open the HTML wrapper, but the plot image inside it did not render because relative asset paths were not being rewritten for the workspace viewer.
3. The user had to explain the failure manually with feedback like "the plot doesn't show!" instead of the system treating that as a product signal automatically.

## What We Learned

### 1. Artifact references are not enough by themselves

If the assistant says `[[artifact:...]]`, the UI should usually do something visible with it. The user should not have to mentally translate a token into a file operation.

### 2. Viewer correctness matters as much as artifact creation

Generating `Matplotlib_Plot.html` was only half the job. If the viewer cannot resolve relative `img` or `link` paths inside that HTML, the user experiences failure even though the workspace contains the "right" file.

### 3. Corrective user replies are product telemetry

Replies like "the plot doesn't show", "this didn't have enough context", and "the artifact isn't interactive" should be treated as failure signals for the previous assistant turn, not just as ordinary follow-up chat.

### 4. Helper prose often adds clutter instead of clarity

Phrases like "You can view the task board here: `[[artifact:task-board.md]]`" are weaker than simply surfacing the artifact inline or opening it in the viewer.

### 5. Repo-improvement passes should mine recent chat failures

When the user asks us to improve `ai-chat`, the app should ground that work in recent corrective feedback from `chat.db` instead of relying on memory or vague intuition.

## Concrete Improvements

### A. Treat artifact references as UI affordances, not dead text

The front-end now recognizes helper boilerplate around artifact references, collapses it, and auto-opens the most relevant previewable artifact from the most recent assistant turn.

References:

- [static/app.js#L92-L95](../static/app.js#L92-L95)
- [static/app.js#L2842-L2901](../static/app.js#L2842-L2901)
- [tests/test_frontend_settings_ui.py#L94-L116](../tests/test_frontend_settings_ui.py#L94-L116)

Snippet:

```js
const ARTIFACT_HELPER_TEXT_PATTERNS = [
    /^\s*(?:you can )?(?:open|view|inspect|see|read|preview|find)\b[^.!?]*?(?:here|below)?[:.]?\s*$/i,
    /^\s*(?:saved plan(?: and notes)?|saved progress|task board|feedback digest|full feedback digest|artifact|artifacts?)[:.]?\s*$/i,
];

function maybeAutoOpenReferencedArtifact(msg, rawContent = '') {
    if (!msg || msg.dataset.artifactPreviewAutoOpened === 'true') return;
    const previewPath = extractArtifactReferences(rawContent)
        .map(path => resolveArtifactReferencePath(path))
        .find(path => {
            if (!shouldAutoOpenArtifactPreview(path) || !findWorkspaceFileEntry(path)) return false;
            if (assistantTurnIncludesArtifactPath(path)) return true;
            return extractArtifactReferences(rawContent).length === 1 && isMostRecentAssistantMessage(msg);
        });
    if (!previewPath) return;
    msg.dataset.artifactPreviewAutoOpened = 'true';
    openWorkspaceFile(previewPath);
}
```

Why it matters:

- `[[artifact:Matplotlib_Plot.html]]` now behaves more like "show the result" than "mention a file".
- Helper text such as "You can view the task board here" is collapsed to the artifact itself.
- The workspace viewer can open automatically on desktop, which supports the pair-programming feel we want.

### B. Rewrite local asset references inside HTML previews

The viewer now rewrites relative asset references inside HTML artifacts to the workspace inline-view endpoint before rendering them into the iframe.

References:

- [static/app.js#L4545-L4585](../static/app.js#L4545-L4585)
- [static/app.js#L4933-L4934](../static/app.js#L4933-L4934)
- [tests/test_frontend_settings_ui.py#L110-L116](../tests/test_frontend_settings_ui.py#L110-L116)

Snippet:

```js
function prepareHtmlPreviewContent(content, path) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(String(content || ''), 'text/html');

    selectors.forEach(([selector, attribute]) => {
        doc.querySelectorAll(selector).forEach(node => {
            const original = node.getAttribute(attribute);
            const resolvedPath = resolveWorkspaceAssetPath(path, original);
            if (!resolvedPath) return;
            node.setAttribute(attribute, workspaceFileInlineViewUrl(resolvedPath));
        });
    });

    return `<!DOCTYPE html>\n${doc.documentElement.outerHTML}`;
}
```

Why it matters:

- The plot case was not just a chat-output problem. It was a viewer path-resolution problem.
- Once the HTML preview is made workspace-aware, a generated `plot.png` can actually appear inside `Matplotlib_Plot.html`.

### C. Turn corrective user replies into failure signals

The backend now classifies short corrective replies, marks the immediately previous assistant message as negative feedback, and can mine those signals later during repo-improvement work.

References:

- [app.py#L6407-L6472](../app.py#L6407-L6472)
- [app.py#L6475-L6578](../app.py#L6475-L6578)
- [app.py#L11575-L11603](../app.py#L11575-L11603)
- [tests/test_runtime_permissions.py#L228-L339](../tests/test_runtime_permissions.py#L228-L339)

Snippet:

```python
def apply_implicit_feedback_from_user_reply(conn, conversation_id, user_message_id, content):
    signal = detect_implicit_failure_feedback(content)
    if signal.get("label") != "negative":
        return {}

    cursor = conn.cursor()
    cursor.execute(
        '''SELECT id, content
           FROM messages
           WHERE conversation_id = ? AND role = 'assistant' AND id < ?
           ORDER BY id DESC
           LIMIT 1''',
        (conversation_id, user_message_id),
    )
    row = cursor.fetchone()
    if not row:
        return {}

    assistant_message_id = int(row[0])
    cursor.execute('UPDATE messages SET feedback = ? WHERE id = ?', ('negative', assistant_message_id))
    return {
        **signal,
        "assistant_message_id": str(assistant_message_id),
    }
```

Why it matters:

- The DoorDash case and the plot case are both now machine-readable failure signals.
- The app can use real user dissatisfaction from `chat.db` as dev-loop input instead of waiting for us to remember it manually.

### D. Feed recent product feedback into deep repo-improvement runs

When the user explicitly asks to improve the app using chat history or feedback, deep mode now hydrates the session with recent corrective feedback and writes a durable feedback digest artifact into the workspace.

References:

- [app.py#L5214-L5220](../app.py#L5214-L5220)
- [app.py#L5275-L5294](../app.py#L5275-L5294)
- [app.py#L11575-L11603](../app.py#L11575-L11603)
- [docs/harness.md#L58](../docs/harness.md#L58)
- [docs/architecture.md#L49](../docs/architecture.md#L49)

Snippet:

```python
if session.recent_product_feedback_summary:
    lines.extend([
        "",
        "## Recent Product Feedback",
        session.recent_product_feedback_summary,
        "",
        f"[[artifact:{session.recent_product_feedback_artifact_path}]]",
    ])
```

Why it matters:

- Repo-improvement turns can now carry explicit evidence like "the plot doesn't show" or "the artifact isn't interactive".
- The task board can show both the current plan and the recent product failures that motivated it.

### E. Remove helper prose from saved-progress and task-board surfacing

We also cleaned up the saved-progress path so artifacts appear as bare references instead of being wrapped in filler.

References:

- [app.py#L5403-L5469](../app.py#L5403-L5469)
- [app.py#L5293-L5294](../app.py#L5293-L5294)
- [prompts.py#L18-L19](../prompts.py#L18-L19)
- [tests/test_runtime_permissions.py#L597-L603](../tests/test_runtime_permissions.py#L597-L603)

Snippet:

```python
if task_board_path:
    lines.append(f"[[artifact:{task_board_path}]]")
```

Why it matters:

- This directly addresses outputs like:
  `You can view the task board here: [[artifact:task-board.md]]`
- The model is now told to put artifact references on their own line, and the UI is prepared to surface them as actual viewer actions.

## Secondary Examples From Chat History

### Example: non-interactive artifact failure

From conversation `030cadad-eb0a-4a76-ad77-f4e4d1f2e03b`:

User feedback:

> the artifact isn't interactive and you can't pick one and it's not basedon the actual door dash menu near me and its not a real artifact :(

Lesson:

- "Artifact created" is not enough.
- The artifact has to match the real job the user asked for: interactivity, grounding, and actual usefulness.

### Example: artifact helper prose instead of direct surfacing

From conversation `285201c9-085c-4ee4-9bca-8f6214e671b4`:

Assistant output:

> You can view the task board here: `[[artifact:task-board.md]]`

Lesson:

- This is exactly the sort of sentence the UI should collapse or replace with the artifact presentation itself.
- The more we make artifacts first-class in the UI, the less the model needs to narrate obvious file-view actions in chat.

## Recommended Dev Loop

Based on these failures and fixes, the best loop for improving `ai-chat` is:

1. Run a realistic prompt that asks for a concrete result, not just an explanation.
2. Capture the assistant output and the final one or two user replies from `chat.db`.
3. Treat corrective user replies as failure evidence and updated acceptance criteria.
4. Patch the product at the layer where the failure actually happened:
   UI surfacing, viewer rendering, runtime behavior, or prompt guidance.
5. Add a narrow regression test for the specific failure shape.
6. Re-run the prompt family and compare the new output to the prior failure case.

The key shift is that the user's last reply is not just conversation. It is evaluation data. Once we treat it that way, the app gets better much faster.

## Quick Reference

- Representative prompt/output failure: `data/chat.db`, conversation `343c7b07-ab29-4e2d-a707-4d94795d274a`
- Baseline commit: `020e515`
- Artifact surfacing logic: [static/app.js#L92-L95](../static/app.js#L92-L95), [static/app.js#L2842-L2901](../static/app.js#L2842-L2901)
- HTML preview fix: [static/app.js#L4545-L4585](../static/app.js#L4545-L4585)
- Feedback ingestion: [app.py#L6407-L6578](../app.py#L6407-L6578)
- Feedback hydration in deep mode: [app.py#L11575-L11603](../app.py#L11575-L11603)
- Artifact-only task-board/saved-progress surfacing: [app.py#L5275-L5294](../app.py#L5275-L5294), [app.py#L5403-L5469](../app.py#L5403-L5469)
- Prompt guidance for artifact and feedback handling: [prompts.py#L18-L19](../prompts.py#L18-L19), [prompts.py#L124-L132](../prompts.py#L124-L132)
