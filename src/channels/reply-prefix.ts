import { resolveEffectiveMessagesConfig, resolveIdentityName } from "../agents/identity.js";
import {
  extractShortModelName,
  type ResponsePrefixContext,
} from "../auto-reply/reply/response-prefix-template.js";
import type { GetReplyOptions } from "../auto-reply/types.js";
import type { OpenClawConfig } from "../config/config.js";

type ModelSelectionContext = Parameters<NonNullable<GetReplyOptions["onModelSelected"]>>[0];

export type ReplyPrefixContextBundle = {
  prefixContext: ResponsePrefixContext;
  responsePrefix?: string;
  responsePrefixContextProvider: () => ResponsePrefixContext;
  onModelSelected: (ctx: ModelSelectionContext) => void;
};

export type ReplyPrefixOptions = Pick<
  ReplyPrefixContextBundle,
  "responsePrefix" | "responsePrefixContextProvider" | "onModelSelected"
>;

/**
 * Resolve the primary model ID from config for fallback detection.
 */
function resolvePrimaryModelId(cfg: OpenClawConfig): string | undefined {
  const model = cfg.agents?.defaults?.model;
  if (!model) {
    return undefined;
  }
  if (typeof model === "string") {
    return model;
  }
  return (model as { primary?: string }).primary ?? undefined;
}

/**
 * Check if the selected model matches the primary model.
 */
function isModelPrimary(
  selectedProvider: string,
  selectedModel: string,
  primaryModelId: string,
): boolean {
  const selectedFull = `${selectedProvider}/${selectedModel}`;
  return (
    selectedFull === primaryModelId ||
    selectedModel === primaryModelId ||
    primaryModelId.endsWith(`/${selectedModel}`)
  );
}

export function createReplyPrefixContext(params: {
  cfg: OpenClawConfig;
  agentId: string;
  channel?: string;
  accountId?: string;
}): ReplyPrefixContextBundle {
  const { cfg, agentId } = params;
  const prefixContext: ResponsePrefixContext = {
    identityName: resolveIdentityName(cfg, agentId),
  };

  const messagesConfig = resolveEffectiveMessagesConfig(cfg, agentId, {
    channel: params.channel,
    accountId: params.accountId,
  });
  const configuredPrefix = messagesConfig.responsePrefix;

  // responsePrefixOnFallback: template shown ONLY when a fallback model is used.
  // Uses the same {model}, {provider}, etc. template variables as responsePrefix.
  const fallbackPrefix = (messagesConfig as Record<string, unknown>).responsePrefixOnFallback as
    | string
    | undefined;
  const primaryModelId = resolvePrimaryModelId(cfg);

  const onModelSelected = (ctx: ModelSelectionContext) => {
    // Mutate the object directly instead of reassigning to ensure closures see updates.
    prefixContext.provider = ctx.provider;
    prefixContext.model = extractShortModelName(ctx.model);
    prefixContext.modelFull = `${ctx.provider}/${ctx.model}`;
    prefixContext.thinkingLevel = ctx.thinkLevel ?? "off";

    // If responsePrefixOnFallback is configured, activate it only on fallback.
    // Sets responsePrefixOverride on the context which is read lazily by
    // normalizeReplyPayload via the responsePrefixContextProvider.
    if (fallbackPrefix && primaryModelId) {
      if (!isModelPrimary(ctx.provider, ctx.model, primaryModelId)) {
        prefixContext.responsePrefixOverride = fallbackPrefix;
      } else {
        prefixContext.responsePrefixOverride = undefined;
      }
    }
  };

  return {
    prefixContext,
    responsePrefix: configuredPrefix,
    responsePrefixContextProvider: () => prefixContext,
    onModelSelected,
  };
}

export function createReplyPrefixOptions(params: {
  cfg: OpenClawConfig;
  agentId: string;
  channel?: string;
  accountId?: string;
}): ReplyPrefixOptions {
  const { responsePrefix, responsePrefixContextProvider, onModelSelected } =
    createReplyPrefixContext(params);
  return { responsePrefix, responsePrefixContextProvider, onModelSelected };
}
