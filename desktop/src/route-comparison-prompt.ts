import type { researchSupportGate } from "./case-support";

interface ComparisonState {
  hasCase: boolean;
  hasTargetAnnotations: boolean;
  candidateCount: number;
  availableCount: number;
  support: ReturnType<typeof researchSupportGate>;
  estimatedSupportChosen: boolean;
  readOnly: boolean;
  engineStopped: boolean;
  generating: boolean;
}

/** Explain the next visible action; this never changes search eligibility. */
export function routeComparisonPrompt(state: ComparisonState): {
  title: string;
  description: string;
} {
  if (state.generating)
    return {
      title: "Searching for routes…",
      description: "Candidates will appear here when the current search finishes.",
    };
  if (!state.hasCase)
    return {
      title: "Open a case to begin",
      description:
        "Use Open case or Import MRI to inspect source imaging and target annotations.",
    };
  if (state.candidateCount > 0)
    return state.availableCount > 0
      ? {
          title: "Choose a route to inspect",
          description:
            "Select route A above. Add route B to compare the two alternatives.",
        }
      : {
          title: "No candidates in this category",
          description: "Choose another candidate category above to inspect its routes.",
        };
  if (!state.hasTargetAnnotations)
    return {
      title: "Target annotations are needed",
      description:
        "Open a case with target annotations, or use Import MRI and choose its matching segmentation.",
    };
  if (state.engineStopped)
    return {
      title: "Route search is unavailable",
      description:
        "Reopen the app to reconnect the local engine. The visible imaging remains available for inspection.",
    };
  if (state.readOnly)
    return {
      title: "Route search runs in the Mac app",
      description:
        "This browser preview supports inspection. Open the case in the Mac app to generate routes.",
    };
  if (state.support.blocked)
    return {
      title: "Access support needs review",
      description:
        state.support.reason ??
        "The case needs reviewed research support before route search is available.",
    };
  if (state.support.requiresEstimatedSupport && !state.estimatedSupportChosen)
    return {
      title: "Choose an access assumption",
      description:
        "In Search settings, decide whether to use estimated image support for hypothetical access windows. This does not certify cortical access.",
    };
  return {
    title: "Generate routes to compare",
    description:
      "Use Generate candidate routes, then select alternatives A and B to inspect them together.",
  };
}
