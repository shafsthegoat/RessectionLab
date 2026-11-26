/**
 * Temporary inspection modes shown as chips in the imaging panes.
 *
 * A mode is a display state the user entered and can leave without losing
 * work: its chip names it and its × (or Escape) restores the previous view.
 * Priority orders the chips and decides which mode one Escape press leaves;
 * the highest priority exits first.
 */
export type ViewerModeId =
  | 'expanded-field'
  | 'expanded-pane'
  | 'result-view'
  | 'approach-view'
  | 'source-section'
  | 'vessel-slab'
  | 'corridor-crop'
  | 'box-crop'
  | 'branch-crop'
  | 'cta-cut'
  | 'lesion-overlap'
  | 'tool-preview'
  | 'modeled-replay'
  | 'generated-episode'
  | 'estimate-view'
  | 'prior-view';

/**
 * `always` keeps the original operative-field contract: any Escape restores it.
 * `guarded` ignores Escape that another control already handled or that was
 * typed into a field or dialog. `none` shows the mode without an Escape path
 * (its way back lives in another control).
 */
export type ModeEscape = 'always' | 'guarded' | 'none';

export interface ViewerModeDefinition {
  label: string;
  priority: number;
  escape: ModeEscape;
}

export const VIEWER_MODES: Readonly<Record<ViewerModeId, ViewerModeDefinition>> = {
  'expanded-field': {label: 'Expanded field', priority: 100, escape: 'always'},
  'expanded-pane': {label: 'Expanded view', priority: 90, escape: 'guarded'},
  'result-view': {label: 'Result view', priority: 80, escape: 'guarded'},
  'approach-view': {label: 'Corridor view', priority: 70, escape: 'guarded'},
  'source-section': {label: 'Section', priority: 65, escape: 'guarded'},
  'vessel-slab': {label: 'Vessel slab', priority: 60, escape: 'guarded'},
  'corridor-crop': {label: 'Corridor crop', priority: 55, escape: 'guarded'},
  'box-crop': {label: 'Box crop', priority: 50, escape: 'guarded'},
  'branch-crop': {label: 'Branch crop', priority: 45, escape: 'none'},
  'cta-cut': {label: 'CTA cut', priority: 40, escape: 'guarded'},
  'lesion-overlap': {label: 'Vessels through lesion', priority: 35, escape: 'guarded'},
  'tool-preview': {label: 'Tool preview', priority: 30, escape: 'guarded'},
  'modeled-replay': {label: 'Modeled removal', priority: 20, escape: 'guarded'},
  'generated-episode': {label: 'Generated episode', priority: 20, escape: 'guarded'},
  'estimate-view': {label: 'Estimate view', priority: 20, escape: 'guarded'},
  'prior-view': {label: 'Population prior', priority: 20, escape: 'guarded'},
};

/** An active mode as a pane or the host reports it. */
export interface ViewerMode {
  id: ViewerModeId;
  /** Overrides the registry name, e.g. a more specific generated source. */
  label?: string;
  /** Value or provenance after the name; keep estimate/supplied/simulated wording. */
  detail?: string;
  /** Accessible name of the × button; defaults to "Exit <label>". */
  exitLabel?: string;
  onExit?: () => void;
}

export interface ResolvedViewerMode extends ViewerMode {
  label: string;
  exitLabel: string;
  priority: number;
  escape: ModeEscape;
}

/** "Exit section", "Exit CTA cut": keep acronyms, lower an initial capital. */
export function modeExitLabel(label: string): string {
  const name = label.trim();
  const lowered = /^[A-Z][a-z]/.test(name) ? name[0].toLowerCase() + name.slice(1) : name;
  return `Exit ${lowered}`;
}

/** Merge registry defaults and order chips by priority, keeping report order on ties. */
export function resolveModes(
  modes: ReadonlyArray<ViewerMode | null | undefined | false | 0 | ''>,
): ResolvedViewerMode[] {
  return modes
    .flatMap((mode, index) => {
      if (!mode) return [];
      const definition = VIEWER_MODES[mode.id];
      const label = mode.label ?? definition.label;
      return [{
        index,
        mode: {
          ...mode,
          label,
          exitLabel: mode.exitLabel ?? modeExitLabel(label),
          priority: definition.priority,
          escape: mode.onExit ? definition.escape : 'none' as ModeEscape,
        },
      }];
    })
    .sort((a, b) => b.mode.priority - a.mode.priority || a.index - b.index)
    .map(entry => entry.mode);
}

export interface EscapeContext {
  /** Another control (an open popover, a jog pad) already handled this Escape. */
  defaultPrevented: boolean;
  /** Escape was pressed in a text field, select, editable region or dialog. */
  editableTarget: boolean;
  /** A held key's auto-repeat; it must not leave several modes at once. */
  repeat?: boolean;
  /** The workspace is not on screen (another stage is open). */
  offscreen?: boolean;
}

/** The one mode a single Escape press should leave, or null. */
export function escapeTarget(
  modes: readonly ResolvedViewerMode[],
  context: EscapeContext,
): ResolvedViewerMode | null {
  const blocked = context.defaultPrevented || context.editableTarget || context.repeat === true || context.offscreen === true;
  for (const mode of modes) {
    if (!mode.onExit || mode.escape === 'none') continue;
    if (mode.escape === 'always') return mode;
    if (!blocked) return mode;
  }
  return null;
}
