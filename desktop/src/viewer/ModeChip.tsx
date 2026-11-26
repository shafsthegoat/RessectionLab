import {useEffect, useRef} from 'react';
import type {ReactNode, RefObject} from 'react';
import type {LucideIcon} from 'lucide-react';
import {
  Box,
  Crop,
  Crosshair,
  Eye,
  FlaskConical,
  GitBranch,
  History,
  Layers2,
  Map as MapIcon,
  Maximize2,
  ScanLine,
  SquareDashed,
  Waypoints,
  X,
} from 'lucide-react';
import {escapeTarget} from './viewerModes';
import type {ResolvedViewerMode, ViewerModeId} from './viewerModes';
import './mode-chip.css';

export interface ModeChipProps {
  /** Short mode name in sentence case. */
  label: string;
  /** Value or provenance after the name. */
  detail?: string;
  icon?: LucideIcon;
  /** Accessible name of the × button, e.g. "Exit section". */
  exitLabel?: string;
  onExit?: () => void;
  /** Escape leaves this mode too; adds "(Esc)" to the × tooltip. */
  escapable?: boolean;
  /** Optional inline toggles that belong to the mode (use `.rl-mode-chip-action`). */
  children?: ReactNode;
  modeId?: string;
}

/**
 * A visible, escapable mode: accent outline, icon, name, optional detail and ×.
 * Exported for any pane that enters a temporary inspection state.
 */
export function ModeChip({label, detail, icon: Icon, exitLabel, onExit, escapable, children, modeId}: ModeChipProps) {
  const exitName = exitLabel ?? `Exit ${label}`;
  return (
    <div className="rl-mode-chip" role="group" aria-label={`${label} mode`} data-mode={modeId}>
      {Icon && <Icon className="rl-mode-chip-icon" size={14} aria-hidden="true" />}
      <span className="rl-mode-chip-text">
        <span className="rl-mode-chip-name">{label}</span>
        {detail && <span className="rl-mode-chip-detail">{detail}</span>}
      </span>
      {children}
      {onExit && (
        <button type="button" className="rl-mode-chip-exit" aria-label={exitName}
          title={escapable ? `${exitName} (Esc)` : exitName} onClick={onExit}>
          <X size={14} aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

const MODE_ICONS: Record<ViewerModeId, LucideIcon> = {
  'expanded-field': Maximize2,
  'expanded-pane': Maximize2,
  'result-view': Box,
  'approach-view': Waypoints,
  'source-section': ScanLine,
  'vessel-slab': Layers2,
  'corridor-crop': Crop,
  'box-crop': Crop,
  'branch-crop': GitBranch,
  'cta-cut': ScanLine,
  'lesion-overlap': Eye,
  'tool-preview': Crosshair,
  'modeled-replay': History,
  'generated-episode': FlaskConical,
  'estimate-view': SquareDashed,
  'prior-view': MapIcon,
};

/**
 * The chips of one pane, top-left under its header. Pointer presses on a chip
 * never reach the pane's camera or cursor handlers.
 */
export function ModeChipStack({modes, actions, className}: {
  modes: readonly ResolvedViewerMode[];
  actions?: Partial<Record<ViewerModeId, ReactNode>>;
  className?: string;
}) {
  const stack = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = stack.current;
    if (!element) return;
    // Native listeners: the 3D camera controls listen on the pane element itself.
    const stop = (event: Event) => event.stopPropagation();
    element.addEventListener('pointerdown', stop);
    element.addEventListener('dblclick', stop);
    return () => {
      element.removeEventListener('pointerdown', stop);
      element.removeEventListener('dblclick', stop);
    };
  }, []);
  return (
    <div ref={stack} className={className ? `rl-mode-chips ${className}` : 'rl-mode-chips'}
      aria-live="polite" aria-relevant="additions">
      {modes.map((mode, index) => (
        <ModeChip key={`${mode.id}:${mode.label}:${modes.slice(0, index).filter(other => other.id === mode.id && other.label === mode.label).length}`} modeId={mode.id} icon={MODE_ICONS[mode.id]} label={mode.label}
          detail={mode.detail} exitLabel={mode.exitLabel} onExit={mode.onExit} escapable={mode.escape !== 'none'}>
          {actions?.[mode.id]}
        </ModeChip>
      ))}
    </div>
  );
}

/** Text entry, selects, editable regions and dialogs keep their own Escape. */
export function editableEscapeTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  if (target instanceof HTMLElement && target.isContentEditable) return true;
  return !!target.closest(
    'textarea, select, input:not([type=range]):not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]):not([type=reset]), [role=dialog], [role=alertdialog], dialog[open], [aria-modal=true]',
  );
}

/**
 * One document Escape listener: each press leaves the highest-priority mode.
 * `scope` is the workspace element; while it is hidden only `always` modes respond.
 */
export function useModeEscape(modes: readonly ResolvedViewerMode[], scope?: RefObject<HTMLElement | null>): void {
  const latest = useRef(modes);
  latest.current = modes;
  const listening = modes.some(mode => mode.onExit && mode.escape !== 'none');
  useEffect(() => {
    if (!listening) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      const mode = escapeTarget(latest.current, {
        defaultPrevented: event.defaultPrevented,
        editableTarget: editableEscapeTarget(event.target),
        repeat: event.repeat,
        offscreen: !!scope?.current && scope.current.getClientRects().length === 0,
      });
      if (!mode) return;
      if (mode.escape === 'guarded') event.preventDefault();
      mode.onExit?.();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [listening]);
}

/** Mount once per workspace, with every active mode across its panes. */
export function ModeEscape({modes, scope}: {modes: readonly ResolvedViewerMode[]; scope?: RefObject<HTMLElement | null>}) {
  useModeEscape(modes, scope);
  return null;
}
