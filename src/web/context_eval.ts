import {
    contextEvalReport,
    fixtureReviewDetail,
    fixtureReviewList,
    settingsOverlay,
} from "./dom.js";
import { state } from "./state.js";
import type {
    ContextEvalExampleCase,
    ContextEvalFixtureDetailRecord,
    ContextEvalFixtureDetailResponse,
    ContextEvalFixtureListResponse,
    ContextEvalFixtureReviewResponse,
    ContextEvalPromotionResponse,
    ContextEvalReport,
    ContextEvalTriageBucket,
} from "./types.js";

interface ContextEvalControllerDeps {
    escapeHtml: (value: string) => string;
    fetchJson: <T>(url: string, init?: RequestInit) => Promise<T>;
    formatDecimal: (value?: number | null, digits?: number) => string;
    openFile: (path: string) => Promise<void>;
    setComposerHint: (text: string) => void;
    showSettings: () => void;
}

export interface ContextEvalController {
    loadContextEvalFixtures: () => Promise<void>;
    loadContextEvalReport: () => Promise<void>;
    renderContextEvalReport: () => void;
    renderFixtureReviewDetail: () => void;
    renderFixtureReviewList: () => void;
}

export function createContextEvalController(deps: ContextEvalControllerDeps): ContextEvalController {
    function workspaceRelativeCapturePath(sourcePath: string): string {
        const normalized = String(sourcePath || "").replace(/\\/g, "/");
        const marker = "/.ai/context-evals/";
        const markerIndex = normalized.lastIndexOf(marker);
        if (markerIndex === -1) return "";
        return `.ai/context-evals/${normalized.slice(markerIndex + marker.length)}`;
    }

    function suggestContextEvalFixtureName(bucket: ContextEvalTriageBucket, exampleCase: ContextEvalExampleCase): string {
        const segments = [
            exampleCase.phase || "capture",
            bucket.category || "context_eval",
            bucket.key || exampleCase.name || "case",
        ];
        return segments
            .join("_")
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "_")
            .replace(/^_+|_+$/g, "")
            || "captured_context_eval_case";
    }

    function describeFixtureCoverage(coverage?: ContextEvalTriageBucket["fixture_coverage"] | null): string {
        if (coverage?.accepted_count) return "covered by accepted fixture";
        if (coverage?.candidate_count) return "candidate fixture exists";
        if (coverage?.suggested_count) return "suggested fixture queued";
        return "needs promotion";
    }

    function renderFixtureReviewList(): void {
        if (state.fixtureReviewLoading) {
            fixtureReviewList.innerHTML = `<div class="context-eval-empty">Loading fixtures…</div>`;
            return;
        }

        const selectedFilter = state.fixtureReviewFilter || "all";
        const fixtures = (state.contextEvalFixtures || []).filter((fixture) => (
            selectedFilter === "all" ? true : fixture.review_status === selectedFilter
        ));
        if (!fixtures.length) {
            fixtureReviewList.innerHTML = `<div class="context-eval-empty">No fixtures for this filter yet.</div>`;
            return;
        }

        fixtureReviewList.innerHTML = fixtures.map((fixture) => {
            const fixturePath = workspaceRelativeCapturePath(fixture.path) || fixture.path;
            const sourcePath = workspaceRelativeCapturePath(fixture.source_path) || fixture.source_path;
            return `
                <article class="fixture-review-item${state.selectedFixturePath === fixture.path ? " active" : ""}">
                    <div class="context-eval-focus">
                        <strong>${deps.escapeHtml(fixture.name || "Unnamed fixture")}</strong>
                        <span class="context-eval-severity ${deps.escapeHtml(fixture.review_status || "candidate")}">${deps.escapeHtml(fixture.review_status || "candidate")}</span>
                    </div>
                    <div class="context-eval-meta">
                        <span>${deps.escapeHtml(fixture.source_trigger || "unknown trigger")}</span>
                        <span>${deps.escapeHtml(fixture.updated_at || fixture.promoted_at || "")}</span>
                    </div>
                    <div class="fixture-review-paths">
                        <code>${deps.escapeHtml(fixturePath)}</code>
                        ${sourcePath ? `<code>${deps.escapeHtml(sourcePath)}</code>` : ""}
                    </div>
                    <div class="context-eval-actions">
                        <button type="button" class="context-eval-button" data-fixture-select="${deps.escapeHtml(fixture.path)}">
                            ${state.selectedFixturePath === fixture.path ? "Selected" : "Inspect"}
                        </button>
                        <button type="button" class="context-eval-button" data-fixture-open="${deps.escapeHtml(fixturePath)}">
                            Open
                        </button>
                        <button type="button" class="context-eval-button" data-fixture-status="suggested" data-fixture-path="${deps.escapeHtml(fixture.path)}">
                            Suggested
                        </button>
                        <button type="button" class="context-eval-button" data-fixture-status="candidate" data-fixture-path="${deps.escapeHtml(fixture.path)}">
                            Candidate
                        </button>
                        <button type="button" class="context-eval-button" data-fixture-status="accepted" data-fixture-path="${deps.escapeHtml(fixture.path)}">
                            Accepted
                        </button>
                        <button type="button" class="context-eval-button" data-fixture-status="superseded" data-fixture-path="${deps.escapeHtml(fixture.path)}">
                            Supersede
                        </button>
                    </div>
                </article>
            `;
        }).join("");

        fixtureReviewList.querySelectorAll<HTMLButtonElement>("[data-fixture-select]").forEach((button) => {
            button.addEventListener("click", async () => {
                const fixturePath = button.dataset.fixtureSelect || "";
                if (!fixturePath) return;
                state.selectedFixturePath = fixturePath;
                if (state.compareFixturePath === fixturePath) {
                    state.compareFixturePath = "";
                    state.compareFixtureDetail = null;
                }
                renderFixtureReviewList();
                await loadSelectedFixtureDetails();
            });
        });
        fixtureReviewList.querySelectorAll<HTMLButtonElement>("[data-fixture-open]").forEach((button) => {
            button.addEventListener("click", () => {
                const path = button.dataset.fixtureOpen || "";
                if (!path || path.startsWith("/")) return;
                void deps.openFile(path);
            });
        });
        fixtureReviewList.querySelectorAll<HTMLButtonElement>("[data-fixture-status]").forEach((button) => {
            button.addEventListener("click", async () => {
                const fixturePath = button.dataset.fixturePath || "";
                const reviewStatus = button.dataset.fixtureStatus || "";
                if (!fixturePath || !reviewStatus) return;
                try {
                    const response = await deps.fetchJson<ContextEvalFixtureReviewResponse>("/api/context-evals/fixtures/review", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            fixture_path: fixturePath,
                            review_status: reviewStatus,
                        }),
                    });
                    deps.setComposerHint(`Updated ${response.fixture_name} to ${response.review_status}.`);
                    await loadContextEvalFixtures();
                } catch (error) {
                    deps.setComposerHint(error instanceof Error ? error.message : "Could not update fixture review state.");
                }
            });
        });
    }

    function renderFixtureDetailValue(value: unknown): string {
        if (Array.isArray(value)) {
            if (!value.length) return `<div class="context-eval-empty">None</div>`;
            return `<ul class="context-eval-detail-list">${value.map((item) => `<li>${deps.escapeHtml(String(item))}</li>`).join("")}</ul>`;
        }
        if (value && typeof value === "object") {
            const entries = Object.entries(value as Record<string, unknown>);
            if (!entries.length) return `<div class="context-eval-empty">None</div>`;
            return `
                <div class="fixture-detail-kv-list">
                    ${entries.map(([key, itemValue]) => `
                        <div class="fixture-detail-kv-row">
                            <span class="fixture-detail-kv-key">${deps.escapeHtml(key)}</span>
                            <code>${deps.escapeHtml(JSON.stringify(itemValue))}</code>
                        </div>
                    `).join("")}
                </div>
            `;
        }
        if (value === undefined || value === null || value === "") {
            return `<div class="context-eval-empty">None</div>`;
        }
        return `<code>${deps.escapeHtml(String(value))}</code>`;
    }

    function renderFixtureComparison(
        selectedFixture: ContextEvalFixtureDetailRecord,
        compareFixture: ContextEvalFixtureDetailRecord | null,
    ): string {
        if (!compareFixture) {
            return `<div class="context-eval-empty">Choose a comparison fixture to inspect differences.</div>`;
        }
        const selectedExpectation = JSON.stringify(selectedFixture.payload.expectation || {});
        const compareExpectation = JSON.stringify(compareFixture.payload.expectation || {});
        const selectedCandidates = JSON.stringify(selectedFixture.payload.selection_candidates || []);
        const compareCandidates = JSON.stringify(compareFixture.payload.selection_candidates || []);
        const differences = [
            selectedFixture.review_status !== compareFixture.review_status
                ? `Review status: ${selectedFixture.review_status} vs ${compareFixture.review_status}`
                : "",
            selectedExpectation !== compareExpectation
                ? "Expectation payload differs"
                : "",
            selectedCandidates !== compareCandidates
                ? "Selection candidates differ"
                : "",
            JSON.stringify(selectedFixture.payload.policy_inputs || {}) !== JSON.stringify(compareFixture.payload.policy_inputs || {})
                ? "Policy inputs differ"
                : "",
        ].filter(Boolean);
        return differences.length
            ? `<ul class="context-eval-detail-list">${differences.map((item) => `<li>${deps.escapeHtml(item)}</li>`).join("")}</ul>`
            : `<div class="context-eval-empty">These fixtures currently have matching core payload sections.</div>`;
    }

    function renderFixtureReviewDetail(): void {
        if (state.fixtureDetailLoading) {
            fixtureReviewDetail.innerHTML = `<div class="context-eval-empty">Loading fixture detail…</div>`;
            return;
        }

        const selectedFixture = state.selectedFixtureDetail;
        if (!selectedFixture) {
            fixtureReviewDetail.innerHTML = `<div class="context-eval-empty">Select a fixture to inspect its full payload and compare it with another reviewed case.</div>`;
            return;
        }

        const compareOptions = (state.contextEvalFixtures || [])
            .filter((fixture) => fixture.path !== selectedFixture.path)
            .map((fixture) => `
                <option value="${deps.escapeHtml(fixture.path)}"${fixture.path === state.compareFixturePath ? " selected" : ""}>
                    ${deps.escapeHtml(`${fixture.name} (${fixture.review_status})`)}
                </option>
            `)
            .join("");

        fixtureReviewDetail.innerHTML = `
            <div class="context-eval-card">
                <div class="context-eval-label">Fixture Detail</div>
                <div class="context-eval-focus">
                    <strong>${deps.escapeHtml(selectedFixture.name)}</strong>
                    <span class="context-eval-severity ${deps.escapeHtml(selectedFixture.review_status)}">${deps.escapeHtml(selectedFixture.review_status)}</span>
                </div>
                <div class="context-eval-meta">
                    <span>${deps.escapeHtml(selectedFixture.source_trigger || "unknown trigger")}</span>
                    <span>${deps.escapeHtml(selectedFixture.updated_at || selectedFixture.promoted_at || "")}</span>
                </div>
                <div class="fixture-review-paths">
                    <code>${deps.escapeHtml(workspaceRelativeCapturePath(selectedFixture.path) || selectedFixture.path)}</code>
                    ${selectedFixture.source_path ? `<code>${deps.escapeHtml(workspaceRelativeCapturePath(selectedFixture.source_path) || selectedFixture.source_path)}</code>` : ""}
                </div>
            </div>
            <div class="context-eval-card">
                <div class="settings-section-header">
                    <strong>Compare</strong>
                    <div class="settings-filter-row">
                        <select id="fixtureCompareSelect" aria-label="Compare fixture">
                            <option value="">No comparison</option>
                            ${compareOptions}
                        </select>
                    </div>
                </div>
                ${renderFixtureComparison(selectedFixture, state.compareFixtureDetail)}
            </div>
            <div class="context-eval-card">
                <div class="context-eval-label">Policy Inputs</div>
                ${renderFixtureDetailValue(selectedFixture.payload.policy_inputs)}
            </div>
            <div class="context-eval-card">
                <div class="context-eval-label">Expectation</div>
                ${renderFixtureDetailValue(selectedFixture.payload.expectation)}
            </div>
            <div class="context-eval-card">
                <div class="context-eval-label">Selection Candidates</div>
                ${renderFixtureDetailValue(selectedFixture.payload.selection_candidates)}
            </div>
        `;

        const compareSelect = fixtureReviewDetail.querySelector<HTMLSelectElement>("#fixtureCompareSelect");
        if (compareSelect) {
            compareSelect.addEventListener("change", async () => {
                state.compareFixturePath = compareSelect.value || "";
                await loadSelectedFixtureDetails();
            });
        }
    }

    function renderContextEvalReport(): void {
        if (state.contextEvalLoading) {
            contextEvalReport.innerHTML = `<div class="context-eval-empty">Loading replay report…</div>`;
            return;
        }

        if (state.contextEvalError) {
            contextEvalReport.innerHTML = `<div class="context-eval-empty">${deps.escapeHtml(state.contextEvalError)}</div>`;
            return;
        }

        const report = state.contextEvalReport;
        if (!report || !report.total_cases) {
            contextEvalReport.innerHTML = `
                <div class="context-eval-empty">
                    No captured replay cases yet. Negative feedback, retries, and corrective follow-ups will appear here.
                </div>
            `;
            return;
        }

        const recommended = report.recommended_fix;
        const topBuckets = report.top_triage_buckets || [];
        const recentFailures = report.recent_failures || [];
        const selectedBucket = topBuckets.find((bucket) => bucket.key === state.selectedContextEvalBucketKey)
            || recommended
            || topBuckets[0]
            || null;
        const exampleCases = selectedBucket?.example_cases || [];
        const selectedExample = exampleCases[
            Math.min(Math.max(state.selectedContextEvalExampleIndex, 0), Math.max(exampleCases.length - 1, 0))
        ] || null;
        const selectedCapturePath = selectedExample ? workspaceRelativeCapturePath(selectedExample.source_path) : "";
        const needsPromotion = Boolean(
            selectedBucket
            && !selectedBucket.fixture_coverage?.accepted_count
            && !selectedBucket.fixture_coverage?.suggested_count
            && !selectedBucket.fixture_coverage?.candidate_count
        );
        const suggestedFixtureName = selectedBucket && selectedExample
            ? suggestContextEvalFixtureName(selectedBucket, selectedExample)
            : "";
        const promotionSuggestion = selectedBucket?.promotion_suggestion || null;
        contextEvalReport.innerHTML = `
            <div class="context-eval-card">
                <div class="context-eval-label">Current Signal</div>
                <div class="context-eval-stats">
                    <div class="context-eval-stat">
                        <strong>${deps.escapeHtml(String(report.total_cases))}</strong>
                        <span>captures</span>
                    </div>
                    <div class="context-eval-stat">
                        <strong>${deps.escapeHtml(String(report.failed_cases))}</strong>
                        <span>failing</span>
                    </div>
                    <div class="context-eval-stat">
                        <strong>${deps.escapeHtml(deps.formatDecimal(report.average_score))}</strong>
                        <span>avg score</span>
                    </div>
                </div>
            </div>
            ${recommended ? `
                <div class="context-eval-card">
                    <div class="context-eval-label">Recommended Next Fix</div>
                    <div class="context-eval-focus">
                        <strong>${deps.escapeHtml(recommended.title)}</strong>
                        <span class="context-eval-severity ${deps.escapeHtml(recommended.severity)}">${deps.escapeHtml(recommended.severity)}</span>
                    </div>
                    <p>${deps.escapeHtml(recommended.recommendation)}</p>
                    <div class="context-eval-meta">
                        <span>${deps.escapeHtml(String(recommended.failure_count))} failures</span>
                        <span>${deps.escapeHtml(String(recommended.case_count))} cases</span>
                        <span>score ${deps.escapeHtml(deps.formatDecimal(recommended.priority_score))}</span>
                    </div>
                    ${recommended.promotion_suggestion?.should_suggest ? `
                        <div class="context-eval-example context-eval-suggestion">
                            <div class="context-eval-label">Suggested Promotion</div>
                            <p>${deps.escapeHtml(recommended.promotion_suggestion.reason)}</p>
                        </div>
                    ` : ""}
                </div>
            ` : ""}
            <div class="context-eval-card">
                <div class="context-eval-label">Top Fixes</div>
                ${topBuckets.length ? `
                    <div class="context-eval-list">
                        ${topBuckets.slice(0, 4).map((bucket) => `
                            <article class="context-eval-item${selectedBucket?.key === bucket.key ? " active" : ""}">
                                <div class="context-eval-focus">
                                    <strong>${deps.escapeHtml(bucket.title)}</strong>
                                    <span class="context-eval-severity ${deps.escapeHtml(bucket.severity)}">${deps.escapeHtml(bucket.severity)}</span>
                                </div>
                                <div class="context-eval-meta">
                                    <span>${deps.escapeHtml(String(bucket.failure_count))} failures</span>
                                    <span>${deps.escapeHtml(String(bucket.case_count))} cases</span>
                                    <span>${describeFixtureCoverage(bucket.fixture_coverage)}</span>
                                    ${bucket.promotion_suggestion?.should_suggest ? "<span>suggested candidate fixture</span>" : ""}
                                </div>
                                <p>${deps.escapeHtml(bucket.recommendation)}</p>
                                <div class="context-eval-actions">
                                    <button
                                        type="button"
                                        class="context-eval-button"
                                        data-context-eval-bucket="${deps.escapeHtml(bucket.key)}"
                                    >
                                        ${selectedBucket?.key === bucket.key ? "Viewing example" : "Open example"}
                                    </button>
                                </div>
                                ${bucket.example_cases[0] ? `
                                    <div class="context-eval-example">
                                        <div class="context-eval-label">Example</div>
                                        <div>${deps.escapeHtml(bucket.example_cases[0].name)}</div>
                                        <div class="context-eval-meta">
                                            <span>${deps.escapeHtml(bucket.example_cases[0].phase)}</span>
                                            <span>${deps.escapeHtml(bucket.example_cases[0].trigger)}</span>
                                        </div>
                                    </div>
                                ` : ""}
                            </article>
                        `).join("")}
                    </div>
                ` : `<div class="context-eval-empty">No triage buckets yet.</div>`}
            </div>
            ${selectedBucket && selectedExample ? `
                <div class="context-eval-card">
                    <div class="context-eval-label">Capture Drill-Down</div>
                    <div class="context-eval-focus">
                        <strong>${deps.escapeHtml(selectedBucket.title)}</strong>
                        <span class="context-eval-severity ${deps.escapeHtml(selectedBucket.severity)}">${deps.escapeHtml(selectedBucket.severity)}</span>
                    </div>
                    <div class="context-eval-meta">
                        <span>${deps.escapeHtml(selectedExample.name)}</span>
                        <span>${deps.escapeHtml(selectedExample.phase)}</span>
                        <span>${deps.escapeHtml(selectedExample.trigger)}</span>
                    </div>
                    <div class="context-eval-meta">
                        <span>${selectedBucket.fixture_coverage?.accepted_count ? "Accepted fixture exists" : "No accepted fixture yet"}</span>
                        ${selectedBucket.fixture_coverage?.suggested_count ? `<span>${deps.escapeHtml(String(selectedBucket.fixture_coverage.suggested_count))} suggested fixture(s)</span>` : ""}
                        ${selectedBucket.fixture_coverage?.candidate_count ? `<span>${deps.escapeHtml(String(selectedBucket.fixture_coverage.candidate_count))} candidate fixture(s)</span>` : ""}
                        ${selectedBucket.fixture_coverage?.superseded_count ? `<span>${deps.escapeHtml(String(selectedBucket.fixture_coverage.superseded_count))} superseded</span>` : ""}
                    </div>
                    ${selectedBucket.fixture_coverage?.fixtures?.length ? `
                        <div class="context-eval-example">
                            <div class="context-eval-label">Fixture Coverage</div>
                            <div class="context-eval-list">
                                ${selectedBucket.fixture_coverage.fixtures.map((fixture) => `
                                    <article class="context-eval-item">
                                        <strong>${deps.escapeHtml(fixture.name)}</strong>
                                        <div class="context-eval-meta">
                                            <span>${deps.escapeHtml(fixture.review_status)}</span>
                                        </div>
                                    </article>
                                `).join("")}
                            </div>
                        </div>
                    ` : ""}
                    ${promotionSuggestion?.should_suggest ? `
                        <div class="context-eval-example context-eval-suggestion">
                            <div class="context-eval-label">Suggested Promotion</div>
                            <p>${deps.escapeHtml(promotionSuggestion.reason)}</p>
                            <div class="context-eval-meta">
                                <span>Draft name: ${deps.escapeHtml(suggestedFixtureName || selectedExample.name)}</span>
                                <span>${deps.escapeHtml(promotionSuggestion.suggested_review_status)}</span>
                            </div>
                        </div>
                    ` : ""}
                    ${exampleCases.length > 1 ? `
                        <div class="context-eval-actions">
                            ${exampleCases.map((exampleCase, index) => `
                                <button
                                    type="button"
                                    class="context-eval-button${selectedExample === exampleCase ? " active" : ""}"
                                    data-context-eval-bucket-example="${deps.escapeHtml(String(index))}"
                                >
                                    Example ${deps.escapeHtml(String(index + 1))}
                                </button>
                            `).join("")}
                        </div>
                    ` : ""}
                    <div class="context-eval-kv">
                        <span class="context-eval-kv-label">Replay file</span>
                        <code>${deps.escapeHtml(selectedCapturePath || selectedExample.source_path)}</code>
                    </div>
                    <div class="context-eval-actions">
                        <button
                            type="button"
                            class="context-eval-button"
                            data-context-eval-open-capture="true"
                            ${selectedCapturePath ? "" : "disabled"}
                        >
                            Open Capture File
                        </button>
                        <button
                            type="button"
                            class="context-eval-button"
                            data-context-eval-promote-capture="true"
                        >
                            Promote to Fixture
                        </button>
                        ${needsPromotion ? `
                            <button
                                type="button"
                                class="context-eval-button active"
                                data-context-eval-promote-candidate="true"
                            >
                                Promote as Candidate
                            </button>
                        ` : ""}
                    </div>
                    ${selectedBucket.fixture_coverage?.fixtures?.length ? `
                        <div class="context-eval-actions">
                            ${selectedBucket.fixture_coverage.fixtures.map((fixture) => `
                                <button
                                    type="button"
                                    class="context-eval-button"
                                    data-context-eval-open-fixture="${deps.escapeHtml(fixture.path)}"
                                >
                                    Inspect ${deps.escapeHtml(fixture.review_status)} fixture
                                </button>
                            `).join("")}
                        </div>
                    ` : ""}
                    <div class="context-eval-drill-grid">
                        <div class="context-eval-drill-panel">
                            <div class="context-eval-label">Failed Checks</div>
                            ${(selectedExample.failed_checks || []).length ? `
                                <ul class="context-eval-detail-list">
                                    ${(selectedExample.failed_checks || []).map((failedCheck) => `
                                        <li>${deps.escapeHtml(failedCheck)}</li>
                                    `).join("")}
                                </ul>
                            ` : `<div class="context-eval-empty">No failed checks recorded.</div>`}
                        </div>
                        <div class="context-eval-drill-panel">
                            <div class="context-eval-label">Selected Keys</div>
                            ${(selectedExample.selected_keys || []).length ? `
                                <ul class="context-eval-detail-list">
                                    ${(selectedExample.selected_keys || []).map((selectedKey) => `
                                        <li><code>${deps.escapeHtml(selectedKey)}</code></li>
                                    `).join("")}
                                </ul>
                            ` : `<div class="context-eval-empty">No selected keys recorded.</div>`}
                        </div>
                    </div>
                </div>
            ` : ""}
            <div class="context-eval-card">
                <div class="context-eval-label">Recent Failures</div>
                ${recentFailures.length ? `
                    <div class="context-eval-list">
                        ${recentFailures.slice(0, 3).map((failure) => `
                            <article class="context-eval-item">
                                <strong>${deps.escapeHtml(failure.name)}</strong>
                                <div class="context-eval-meta">
                                    <span>${deps.escapeHtml(failure.trigger || "unknown")}</span>
                                    <span>score ${deps.escapeHtml(deps.formatDecimal(failure.score))}</span>
                                </div>
                                <p>${deps.escapeHtml((failure.failed_checks || []).slice(0, 2).join(" | "))}</p>
                            </article>
                        `).join("")}
                    </div>
                ` : `<div class="context-eval-empty">No recent failures.</div>`}
            </div>
        `;

        contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-bucket]").forEach((button) => {
            button.addEventListener("click", () => {
                const bucketKey = button.dataset.contextEvalBucket || "";
                if (!bucketKey) return;
                state.selectedContextEvalBucketKey = bucketKey;
                state.selectedContextEvalExampleIndex = 0;
                renderContextEvalReport();
            });
        });
        contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-bucket-example]").forEach((button) => {
            button.addEventListener("click", () => {
                const rawIndex = button.dataset.contextEvalBucketExample || "0";
                const nextIndex = Number.parseInt(rawIndex, 10);
                state.selectedContextEvalExampleIndex = Number.isFinite(nextIndex) ? Math.max(0, nextIndex) : 0;
                renderContextEvalReport();
            });
        });
        contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-open-capture]").forEach((button) => {
            button.addEventListener("click", () => {
                if (!selectedCapturePath) return;
                void deps.openFile(selectedCapturePath);
            });
        });
        contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-promote-capture]").forEach((button) => {
            button.addEventListener("click", async () => {
                if (!selectedExample?.source_path) return;
                const suggestedName = selectedBucket
                    ? suggestContextEvalFixtureName(selectedBucket, selectedExample)
                    : selectedExample.name;
                const fixtureName = window.prompt("Fixture name", suggestedName);
                if (!fixtureName || !fixtureName.trim()) return;
                const reviewStatus = window.prompt("Review status (suggested, candidate, accepted, superseded)", "candidate");
                if (!reviewStatus || !reviewStatus.trim()) return;
                try {
                    const response = await promoteContextEvalCapture(
                        selectedExample.source_path,
                        fixtureName.trim(),
                        reviewStatus.trim().toLowerCase(),
                    );
                    deps.setComposerHint(`Promoted replay case as ${response.fixture_name} (${response.review_status}).`);
                } catch (error) {
                    deps.setComposerHint(error instanceof Error ? error.message : "Could not promote replay case.");
                }
            });
        });
        contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-promote-candidate]").forEach((button) => {
            button.addEventListener("click", async () => {
                if (!selectedExample?.source_path || !selectedBucket) return;
                const fixtureName = suggestContextEvalFixtureName(selectedBucket, selectedExample);
                try {
                    const response = await promoteContextEvalCapture(
                        selectedExample.source_path,
                        fixtureName,
                        "candidate",
                    );
                    deps.setComposerHint(`Promoted ${response.fixture_name} as candidate.`);
                } catch (error) {
                    deps.setComposerHint(error instanceof Error ? error.message : "Could not promote candidate fixture.");
                }
            });
        });
        contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-open-fixture]").forEach((button) => {
            button.addEventListener("click", async () => {
                const fixturePath = button.dataset.contextEvalOpenFixture || "";
                if (!fixturePath) return;
                await inspectFixtureFromTriage(fixturePath);
            });
        });
    }

    async function promoteContextEvalCapture(
        sourcePath: string,
        fixtureName: string,
        reviewStatus: string,
    ): Promise<ContextEvalPromotionResponse> {
        const response = await deps.fetchJson<ContextEvalPromotionResponse>("/api/context-evals/promote", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_path: sourcePath,
                fixture_name: fixtureName,
                review_status: reviewStatus,
            }),
        });
        await loadContextEvalFixtures();
        state.selectedFixturePath = response.fixture_path;
        await loadSelectedFixtureDetails();
        await loadContextEvalReport();
        return response;
    }

    async function autoDraftContextEvalFixture(
        sourcePath: string,
        bucketKey: string,
        fixtureName: string,
    ): Promise<ContextEvalPromotionResponse> {
        return deps.fetchJson<ContextEvalPromotionResponse>("/api/context-evals/auto-draft", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_path: sourcePath,
                bucket_key: bucketKey,
                fixture_name: fixtureName,
            }),
        });
    }

    async function inspectFixtureFromTriage(fixturePath: string): Promise<void> {
        state.selectedFixturePath = fixturePath;
        state.compareFixturePath = "";
        await loadContextEvalFixtures();
        await loadSelectedFixtureDetails();
        deps.showSettings();
    }

    async function loadContextEvalReport(): Promise<void> {
        if (!state.currentWorkspaceId) {
            state.contextEvalReport = null;
            state.contextEvalError = "";
            state.contextEvalLoading = false;
            renderContextEvalReport();
            return;
        }

        state.contextEvalLoading = true;
        state.contextEvalError = "";
        renderContextEvalReport();

        const conversationId = state.conversations.some((conversation) => conversation.id === state.currentConversationId)
            ? state.currentConversationId
            : "";
        const params = new URLSearchParams({
            limit: "100",
            ...(conversationId ? { conversation_id: conversationId } : { workspace_id: state.currentWorkspaceId }),
        });
        const reportUrl = `/api/context-evals/report?${params.toString()}`;

        try {
            let report = await deps.fetchJson<ContextEvalReport>(reportUrl);
            const suggestionTargets = (report.top_triage_buckets || [])
                .filter((bucket) => (
                    bucket.promotion_suggestion?.should_suggest
                    && !bucket.fixture_coverage?.accepted_count
                    && !bucket.fixture_coverage?.suggested_count
                    && !bucket.fixture_coverage?.candidate_count
                ))
                .slice(0, 3);
            let createdAnySuggestions = false;
            for (const bucket of suggestionTargets) {
                const exampleCase = bucket.example_cases?.[0];
                if (!exampleCase?.source_path) continue;
                const response = await autoDraftContextEvalFixture(
                    exampleCase.source_path,
                    bucket.key,
                    "",
                );
                createdAnySuggestions = Boolean(response.created) || createdAnySuggestions;
            }
            if (createdAnySuggestions) {
                report = await deps.fetchJson<ContextEvalReport>(reportUrl);
                if (!settingsOverlay.hidden) {
                    await loadContextEvalFixtures();
                }
            }
            state.contextEvalReport = report;
            const topBucket = state.contextEvalReport.top_triage_buckets?.[0];
            const selectedStillExists = state.contextEvalReport.top_triage_buckets?.some(
                (bucket) => bucket.key === state.selectedContextEvalBucketKey
            );
            state.selectedContextEvalBucketKey = selectedStillExists
                ? state.selectedContextEvalBucketKey
                : (topBucket?.key || "");
            state.selectedContextEvalExampleIndex = 0;
        } catch (error) {
            state.contextEvalReport = null;
            state.contextEvalError = error instanceof Error ? error.message : "Could not load replay report.";
            state.selectedContextEvalBucketKey = "";
            state.selectedContextEvalExampleIndex = 0;
        } finally {
            state.contextEvalLoading = false;
            renderContextEvalReport();
        }
    }

    async function loadContextEvalFixtures(): Promise<void> {
        state.fixtureReviewLoading = true;
        renderFixtureReviewList();
        try {
            const payload = await deps.fetchJson<ContextEvalFixtureListResponse>("/api/context-evals/fixtures");
            state.contextEvalFixtures = payload.fixtures || [];
            const selectedStillExists = state.contextEvalFixtures.some((fixture) => fixture.path === state.selectedFixturePath);
            if (!selectedStillExists) {
                state.selectedFixturePath = state.contextEvalFixtures[0]?.path || "";
                state.compareFixturePath = "";
                state.selectedFixtureDetail = null;
                state.compareFixtureDetail = null;
            }
        } catch (error) {
            state.contextEvalFixtures = [];
            deps.setComposerHint(error instanceof Error ? error.message : "Could not load promoted fixtures.");
        } finally {
            state.fixtureReviewLoading = false;
            renderFixtureReviewList();
            void loadSelectedFixtureDetails();
        }
    }

    async function loadSelectedFixtureDetails(): Promise<void> {
        if (!state.selectedFixturePath) {
            state.selectedFixtureDetail = null;
            state.compareFixtureDetail = null;
            state.fixtureDetailLoading = false;
            renderFixtureReviewDetail();
            return;
        }

        state.fixtureDetailLoading = true;
        renderFixtureReviewDetail();
        try {
            state.selectedFixtureDetail = await deps.fetchJson<ContextEvalFixtureDetailResponse>(
                `/api/context-evals/fixtures/detail?fixture_path=${encodeURIComponent(state.selectedFixturePath)}`
            );
            if (state.compareFixturePath) {
                state.compareFixtureDetail = await deps.fetchJson<ContextEvalFixtureDetailResponse>(
                    `/api/context-evals/fixtures/detail?fixture_path=${encodeURIComponent(state.compareFixturePath)}`
                );
            } else {
                state.compareFixtureDetail = null;
            }
        } catch (error) {
            state.selectedFixtureDetail = null;
            state.compareFixtureDetail = null;
            deps.setComposerHint(error instanceof Error ? error.message : "Could not load fixture detail.");
        } finally {
            state.fixtureDetailLoading = false;
            renderFixtureReviewDetail();
        }
    }

    return {
        loadContextEvalFixtures,
        loadContextEvalReport,
        renderContextEvalReport,
        renderFixtureReviewDetail,
        renderFixtureReviewList,
    };
}
