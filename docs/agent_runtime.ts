export type RunState =
  | "received"
  | "inspect"
  | "replan"
  | "execute_step"
  | "verify_step"
  | "audit_scope"
  | "finalize"
  | "blocked"
  | "failed";

export type StepStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "verified"
  | "blocked"
  | "failed";

export type VerificationResult = "pass" | "fail" | "skipped";
export type ScopeAuditStatus = "complete" | "partial" | "blocked";

export interface AgentRequest {
  raw: string;
  goal: string;
  deliverables: string[];
  verificationTargets: string[];
}

export interface Assumption {
  text: string;
  confidence: number;
  validated: boolean;
}

export interface RunContext {
  workspaceSummary: string;
  repoStatus: string;
  assumptions: Assumption[];
  constraints: string[];
}

export interface PlanStep {
  id: string;
  title: string;
  status: StepStatus;
  dependsOn: string[];
  expectedOutputs: string[];
  executionActions: string[];
  verificationChecks: string[];
  fallback: string;
  evidence: string[];
}

export interface ExecutionPlan {
  version: number;
  steps: PlanStep[];
}

export interface ArtifactRecord {
  path: string;
  kind: "file" | "dir" | "command_output";
  status: "created" | "updated" | "observed" | "missing";
}

export interface VerificationRecord {
  name: string;
  target: string;
  method: string;
  result: VerificationResult;
  evidence: string;
}

export interface ScopeAuditItem {
  name: string;
  requested: boolean;
  present: boolean;
  verified: boolean;
  notes: string;
}

export interface ScopeAudit {
  status: ScopeAuditStatus;
  items: ScopeAuditItem[];
}

export interface RuntimeErrorRecord {
  type: "tool" | "validation" | "planning" | "verification";
  message: string;
  retryable: boolean;
  replanRequired: boolean;
}

export interface AgentRun {
  runId: string;
  request: AgentRequest;
  state: RunState;
  context: RunContext;
  plan: ExecutionPlan;
  artifacts: ArtifactRecord[];
  verification: VerificationRecord[];
  scopeAudit: ScopeAudit;
  errors: RuntimeErrorRecord[];
}

export interface InspectionResult {
  workspaceSummary: string;
  repoStatus: string;
  assumptions?: Assumption[];
  constraints?: string[];
}

export interface StepExecutionResult {
  ok: boolean;
  summary: string;
  artifacts?: ArtifactRecord[];
  evidence?: string[];
  error?: RuntimeErrorRecord;
}

export interface StepVerificationResult {
  passed: boolean;
  records: VerificationRecord[];
  error?: RuntimeErrorRecord;
}

export interface OrchestratorHooks {
  inspect(request: AgentRequest): Promise<InspectionResult>;
  buildPlan(request: AgentRequest, context: RunContext, previousRun: AgentRun): Promise<ExecutionPlan>;
  executeStep(step: PlanStep, run: AgentRun): Promise<StepExecutionResult>;
  verifyStep(step: PlanStep, run: AgentRun): Promise<StepVerificationResult>;
  emitStatus?(run: AgentRun, message: string): Promise<void> | void;
  shouldContinue?(run: AgentRun): boolean;
  maxReplans?: number;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function cloneStep(step: PlanStep): PlanStep {
  return {
    ...step,
    dependsOn: [...step.dependsOn],
    expectedOutputs: [...step.expectedOutputs],
    executionActions: [...step.executionActions],
    verificationChecks: [...step.verificationChecks],
    evidence: [...step.evidence],
  };
}

export function createInitialRun(request: AgentRequest): AgentRun {
  return {
    runId: `run_${Date.now().toString(36)}`,
    request,
    state: "received",
    context: {
      workspaceSummary: "",
      repoStatus: "",
      assumptions: [],
      constraints: [],
    },
    plan: {
      version: 1,
      steps: [],
    },
    artifacts: [],
    verification: [],
    scopeAudit: {
      status: "partial",
      items: [],
    },
    errors: [],
  };
}

export function validatePlan(plan: ExecutionPlan): void {
  assert(plan.steps.length > 0, "Plan must include at least one step.");
  for (const step of plan.steps) {
    assert(step.title.trim().length > 0, `Step ${step.id} is missing a title.`);
    assert(step.expectedOutputs.length > 0, `Step ${step.id} must declare expected outputs.`);
    assert(step.verificationChecks.length > 0, `Step ${step.id} must declare verification checks.`);
  }
}

export function pickNextStep(plan: ExecutionPlan): PlanStep | undefined {
  return plan.steps.find((step) => step.status === "pending");
}

export function auditScope(run: AgentRun): ScopeAudit {
  const items = run.request.deliverables.map((name) => {
    const step = run.plan.steps.find((candidate) =>
      candidate.expectedOutputs.some((output) => output.includes(name)),
    );
    const present = run.artifacts.some((artifact) => artifact.path.includes(name) && artifact.status !== "missing");
    const verified = run.verification.some((record) => record.target.includes(name) && record.result === "pass");
    return {
      name,
      requested: true,
      present: present || Boolean(step && (step.status === "completed" || step.status === "verified")),
      verified: verified || Boolean(step && step.status === "verified"),
      notes: step ? `Mapped to step "${step.title}".` : "No step claimed this deliverable.",
    };
  });

  const status: ScopeAuditStatus = items.every((item) => item.present && item.verified)
    ? "complete"
    : items.some((item) => item.present || item.verified)
      ? "partial"
      : "blocked";

  return { status, items };
}

export class AgentOrchestrator {
  private readonly hooks: OrchestratorHooks;

  constructor(hooks: OrchestratorHooks) {
    this.hooks = hooks;
  }

  async run(request: AgentRequest): Promise<AgentRun> {
    const run = createInitialRun(request);
    const maxReplans = this.hooks.maxReplans ?? 3;
    let replans = 0;

    while (true) {
      if (run.state === "received") {
        run.state = "inspect";
        await this.emit(run, "Inspecting workspace...");
        const inspection = await this.hooks.inspect(request);
        run.context = {
          workspaceSummary: inspection.workspaceSummary,
          repoStatus: inspection.repoStatus,
          assumptions: inspection.assumptions ?? [],
          constraints: inspection.constraints ?? [],
        };
        run.state = "replan";
      }

      if (run.state === "replan") {
        await this.emit(run, "Planning from inspected reality...");
        run.plan = await this.hooks.buildPlan(request, run.context, run);
        validatePlan(run.plan);
        run.plan.steps = run.plan.steps.map((step) => cloneStep(step));
        run.state = "execute_step";
      }

      if (run.state === "execute_step") {
        const nextStep = pickNextStep(run.plan);
        if (!nextStep) {
          run.state = "audit_scope";
        } else {
          nextStep.status = "in_progress";
          await this.emit(run, `Executing step: ${nextStep.title}`);
          const execution = await this.hooks.executeStep(nextStep, run);

          if (execution.artifacts?.length) {
            run.artifacts.push(...execution.artifacts);
          }
          if (execution.evidence?.length) {
            nextStep.evidence.push(...execution.evidence);
          }

          if (!execution.ok) {
            const error = execution.error ?? {
              type: "tool",
              message: execution.summary || `Execution failed for step ${nextStep.id}.`,
              retryable: false,
              replanRequired: true,
            };
            run.errors.push(error);
            nextStep.status = "failed";
            if (error.replanRequired && replans < maxReplans) {
              replans += 1;
              run.state = "replan";
              continue;
            }
            run.state = "audit_scope";
            continue;
          }

          nextStep.status = "completed";
          nextStep.evidence.push(execution.summary);
          run.state = "verify_step";
        }
      }

      if (run.state === "verify_step") {
        const step = run.plan.steps.find((candidate) => candidate.status === "completed");
        if (!step) {
          run.state = "audit_scope";
        } else {
          await this.emit(run, `Verifying step: ${step.title}`);
          const verification = await this.hooks.verifyStep(step, run);
          run.verification.push(...verification.records);

          if (!verification.passed) {
            const error = verification.error ?? {
              type: "verification",
              message: `Verification failed for step ${step.id}.`,
              retryable: false,
              replanRequired: true,
            };
            run.errors.push(error);
            step.status = "failed";
            if (error.replanRequired && replans < maxReplans) {
              replans += 1;
              run.state = "replan";
              continue;
            }
            run.state = "audit_scope";
            continue;
          }

          step.status = "verified";
          run.state = "execute_step";
        }
      }

      if (run.state === "audit_scope") {
        await this.emit(run, "Auditing requested deliverables against verified outputs...");
        run.scopeAudit = auditScope(run);
        const shouldContinue = this.hooks.shouldContinue?.(run) ?? false;

        if (run.scopeAudit.status !== "complete" && shouldContinue && replans < maxReplans) {
          replans += 1;
          run.state = "replan";
          continue;
        }

        run.state = "finalize";
      }

      if (run.state === "finalize") {
        return run;
      }

      if (run.state === "blocked" || run.state === "failed") {
        return run;
      }
    }
  }

  private async emit(run: AgentRun, message: string): Promise<void> {
    await this.hooks.emitStatus?.(run, this.guardStatusMessage(run, message));
  }

  private guardStatusMessage(run: AgentRun, message: string): string {
    if (message.startsWith("Verifying step:")) {
      const step = run.plan.steps.find((candidate) => candidate.status === "completed");
      assert(step, "Cannot emit verification status without a completed step.");
    }
    if (message.startsWith("Executing step:")) {
      const step = pickNextStep(run.plan);
      assert(step, "Cannot emit execution status without a pending step.");
    }
    return message;
  }
}

export const bootstrapSaaSPlan: ExecutionPlan = {
  version: 1,
  steps: [
    {
      id: "inspect-and-scope",
      title: "Inspect workspace and align scope with current repo reality",
      status: "pending",
      dependsOn: [],
      expectedOutputs: ["docs/architecture.md"],
      executionActions: [
        "Summarize existing files and constraints.",
        "Rewrite the plan if the repo shape differs from assumptions.",
      ],
      verificationChecks: [
        "Confirm the plan references the actual repo layout.",
      ],
      fallback: "If the repo is empty, switch to a scaffold-first plan.",
      evidence: [],
    },
    {
      id: "implement-core",
      title: "Implement the smallest complete vertical slice",
      status: "pending",
      dependsOn: ["inspect-and-scope"],
      expectedOutputs: ["app/main.py", "requirements.txt"],
      executionActions: [
        "Create or update app entrypoint.",
        "Add only the dependencies needed for the verified slice.",
      ],
      verificationChecks: [
        "Import or launch the app.",
        "Smoke-test at least one endpoint.",
      ],
      fallback: "If dependencies are unavailable, verify import-level correctness and report the gap.",
      evidence: [],
    },
  ],
};
