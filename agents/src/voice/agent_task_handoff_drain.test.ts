// SPDX-FileCopyrightText: 2026 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import { expect, it, vi } from 'vitest';
import { FunctionCall } from '../llm/chat_context.js';
import { ToolError, handoff, tool } from '../llm/tool_context.js';
import { Future } from '../utils.js';
import { Agent, AgentTask } from './agent.js';
import { AgentSession } from './agent_session.js';
import { FakeLLM } from './testing/fake_llm.js';

// The two turns must not reuse FakeLLM's default call ID while a tool is still running.
class UniqueCallLLM extends FakeLLM {
  private generation = 0;

  override chat(options: Parameters<FakeLLM['chat']>[0]) {
    const stream = super.chat(options);
    const generation = ++this.generation;
    const next = stream.next.bind(stream);
    stream.next = async () => {
      const result = await next();
      if (!result.done && result.value.delta?.toolCalls) {
        result.value.delta.toolCalls = result.value.delta.toolCalls.map((call) =>
          FunctionCall.create({ ...call, callId: `${generation}_${call.callId}` }),
        );
      }
      return result;
    };
    return stream;
  }
}

it.each([
  { timing: 'before handoff', shutdown: false },
  { timing: 'during handoff onExit', shutdown: false },
  { timing: 'during handoff drain', shutdown: false },
  { timing: 'during handoff onExit', shutdown: true },
  { timing: 'during handoff drain', shutdown: true },
] as const)(
  'settles a non-cancellable tool that starts an AgentTask $timing (shutdown=$shutdown)',
  async ({ timing, shutdown }) => {
    const admission = new Future<void>();
    const started = new Future<void>();
    const result = new Future<unknown>();
    const exiting = new Future<void>();
    const finishExit = new Future<void>();
    const finishTool = new Future<void>();
    const taskEntered = vi.fn();
    const targetEntered = vi.fn();
    const target = Agent.create({ instructions: 'target', onEnter: targetEntered });
    const task = AgentTask.create<string>({
      instructions: 'complete immediately',
      onEnter: async () => {
        taskEntered();
        task.complete('done');
      },
    });
    const agent = Agent.create({
      instructions: 'source',
      onExit: async () => {
        exiting.resolve();
        if (timing === 'during handoff onExit') await finishExit.await;
      },
      tools: [
        tool({
          name: 'transfer',
          description: 'Run an inline task after admission.',
          execute: async (_args, { ctx }) => {
            ctx.speechHandle.allowInterruptions = false;
            started.resolve();
            await admission.await;
            try {
              const value = await task.run();
              result.resolve(value);
              return value;
            } catch (error) {
              result.resolve(error);
              if (shutdown) await finishTool.await;
              throw error;
            }
          },
        }),
        tool({
          name: 'switch',
          description: 'Hand off to the target.',
          execute: async () => handoff({ agent: target }),
        }),
      ],
    });
    const session = new AgentSession({
      llm: new UniqueCallLLM([
        { input: 'transfer', toolCalls: [{ name: 'transfer', args: {} }] },
        { input: 'switch', toolCalls: [{ name: 'switch', args: {} }] },
      ]),
      turnHandling: { turnDetection: 'manual' },
    });
    let closing: Promise<void> | undefined;

    try {
      await session.start({ agent });
      const sourceActivity = agent._agentActivity!;
      session.generateReply({ userInput: 'transfer' });
      await started.await;
      await vi.waitFor(() => expect(sourceActivity.currentSpeech).toBeUndefined());

      if (timing === 'before handoff') {
        admission.resolve();
        expect(await result.await).toBe('done');
      }

      session.generateReply({ userInput: 'switch' });
      if (timing !== 'before handoff') {
        if (timing === 'during handoff onExit') {
          await exiting.await;
          expect(sourceActivity.schedulingPaused).toBe(false);
        } else {
          await vi.waitFor(() => expect(sourceActivity.schedulingPaused).toBe(true));
        }
        admission.resolve();
        expect(await result.await).toBeInstanceOf(ToolError);
        expect(await result.await).toHaveProperty(
          'message',
          'the activity that awaited the inline task is draining',
        );
        expect(taskEntered).not.toHaveBeenCalled();
        if (shutdown) closing = session.close();
        finishTool.resolve();
        finishExit.resolve();
      } else {
        expect(taskEntered).toHaveBeenCalledOnce();
      }

      if (shutdown) {
        await closing;
        expect(targetEntered).not.toHaveBeenCalled();
      } else {
        await vi.waitFor(() => expect(targetEntered).toHaveBeenCalledOnce());
        expect(session.currentAgent).toBe(target);
        await session.close();
      }
      expect(agent._agentActivity).toBeUndefined();
      expect(task._agentActivity).toBeUndefined();
      expect(target._agentActivity).toBeUndefined();
    } finally {
      admission.resolve();
      finishTool.resolve();
      finishExit.resolve();
      await session.close();
    }
  },
);
