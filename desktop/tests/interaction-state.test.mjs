import assert from "node:assert/strict";
import { test } from "node:test";
import { setTimeout as wait } from "node:timers/promises";
import { LatestReplayRequests } from "../src/replay-step-request.ts";
import { isOperationCancelled } from "../src/operation-feedback.ts";

test("IPC-wrapped cancellation stays neutral but a failed cancellation acknowledgement is an error", () => {
  assert.equal(
    isOperationCancelled(
      new Error(
        "Error invoking remote method 'research:replay': Error: Operation cancelled",
      ),
    ),
    true,
  );
  assert.equal(
    isOperationCancelled(new Error("Cancellation request timed out")),
    false,
  );
  assert.equal(isOperationCancelled(new Error("An operation failed")), false);
});
test("rapid replay changes coalesce before any backend request", async () => {
  const calls = [];
  const queue = new LatestReplayRequests(
    async (value) => {
      calls.push(value);
    },
    () => {},
    5,
  );
  for (let i = 0; i < 100; i++) queue.request(i);
  await wait(20);
  assert.deepEqual(calls, [99]);
  queue.invalidate();
});
test("slow replay keeps one active request and only the latest subsequent value; stale result cannot apply", async () => {
  const calls = [],
    applied = [];
  let release;
  let active = 0,
    peak = 0;
  const pending = [];
  const queue = new LatestReplayRequests(
    async (value, current) => {
      active++;
      peak = Math.max(peak, active);
      calls.push(value);
      if (value === 0)
        await new Promise((resolve) => {
          release = resolve;
        });
      if (current()) applied.push(value);
      active--;
    },
    (value) => pending.push(value),
    5,
  );
  queue.request(0);
  await wait(15);
  for (let i = 1; i <= 100; i++) queue.request(i);
  await wait(15);
  assert.deepEqual(calls, [0]);
  assert.equal(active, 1);
  release();
  await wait(20);
  assert.deepEqual(calls, [0, 100]);
  assert.deepEqual(applied, [100]);
  assert.equal(peak, 1);
  assert.equal(pending.at(-1), false);
  queue.invalidate();
});
test("case or route invalidation cancels the latest queued request and withholds an old in-flight result", async () => {
  let release;
  const calls = [],
    applied = [];
  const queue = new LatestReplayRequests(
    async (value, current) => {
      calls.push(value);
      await new Promise((resolve) => {
        release = resolve;
      });
      if (current()) applied.push(value);
    },
    () => {},
    5,
  );
  queue.request(0);
  await wait(15);
  queue.request(1);
  queue.invalidate();
  release();
  await wait(20);
  assert.deepEqual(calls, [0]);
  assert.deepEqual(applied, []);
});
