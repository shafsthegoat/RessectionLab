import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { ArrowUpFromLine, ChevronDown, Files, Plus, X } from "lucide-react";

/** Import only. Accepting a proposal as reviewed working anatomy is a separate contract. */
export function StructuralImportDialog({
  disabled,
  onImport,
}: {
  disabled: boolean;
  onImport: (variant: "main" | "nocsf") => Promise<void>;
}) {
  const [open, setOpen] = useState(false),
    [variant, setVariant] = useState<"main" | "nocsf">("main");
  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <button
          className="quiet-button structural-import-trigger"
          disabled={disabled}
        >
          <Plus size={14} /> Add brain-envelope proposal
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="structural-import-dialog">
          <div className="drawer-heading">
            <span className="eyebrow">ADD SOURCE-BOUND EVIDENCE</span>
            <Dialog.Close
              className="icon-button"
              aria-label="Close proposal import"
            >
              <X size={17} />
            </Dialog.Close>
          </div>
          <div className="proposal-import-icon">
            <Files size={23} />
          </div>
          <Dialog.Title>Import an estimated envelope</Dialog.Title>
          <Dialog.Description>
            Add a recorded extraction result for inspection. The proposal stays
            separate from working anatomy and requires review.
          </Dialog.Description>
          <label className="field-label" htmlFor="proposal-variant">
            Recorded model variant
          </label>
          <div className="select-wrap">
            <select
              id="proposal-variant"
              value={variant}
              onChange={(event) =>
                setVariant(event.target.value as "main" | "nocsf")
              }
            >
              <option value="main">SynthStrip · whole-brain envelope</option>
              <option value="nocsf">SynthStrip · no-CSF envelope</option>
            </select>
            <ChevronDown size={14} />
          </div>
          <div className="proposal-file-list">
            <div>
              <span>1</span>
              <p>Original MRI used by the extraction</p>
            </div>
            <div>
              <span>2</span>
              <p>Predicted native-grid mask</p>
            </div>
            <div>
              <span>3</span>
              <p>Recorded extraction report (JSON)</p>
            </div>
          </div>
          <p className="proposal-import-note">
            Source, model, output and coordinate identities must match.
            Cancelling any file chooser leaves the case unchanged.
          </p>
          <button
            className="primary-button"
            onClick={() => {
              setOpen(false);
              void onImport(variant);
            }}
          >
            <ArrowUpFromLine size={14} /> Choose the three source files
          </button>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
