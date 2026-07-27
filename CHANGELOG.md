# Changelog

## [1.2.0](https://github.com/mrwogu/factory-droid-openai/compare/v1.1.0...v1.2.0) (2026-07-27)


### Features

* multi-arch Docker, Podman docs, unauthenticated warning ([6655828](https://github.com/mrwogu/factory-droid-openai/commit/6655828b6b1b84a708716c015127c021be7cd775))


### Bug Fixes

* handle unset DROID_VERSION in Docker build ([278c1e0](https://github.com/mrwogu/factory-droid-openai/commit/278c1e002f73565522a9b9fe635ea59f91788dc8))

## [1.1.0](https://github.com/mrwogu/factory-droid-openai/compare/v1.0.0...v1.1.0) (2026-07-27)


### Features

* add container image, VS Code BYOK docs, and cross-platform CI ([#11](https://github.com/mrwogu/factory-droid-openai/issues/11)) ([508cb15](https://github.com/mrwogu/factory-droid-openai/commit/508cb15c1e5baae1aed6b94f918cef9479dea474))
* add guarded Droid RPC integration ([#9](https://github.com/mrwogu/factory-droid-openai/issues/9)) ([92ec6c3](https://github.com/mrwogu/factory-droid-openai/commit/92ec6c35a444a2093a464cb72a1dee5cf4ed07db))

## 1.0.0 (2026-07-26)


### Features

* add multimodal attachments and multiple tool calls ([6eedf55](https://github.com/mrwogu/factory-droid-openai/commit/6eedf556d075d9784a58cf6455af2c7ddb0d365d))
* add OpenAI-compatible chat completions bridge ([c5aa704](https://github.com/mrwogu/factory-droid-openai/commit/c5aa70440871dc96a80ca515565ad354bcfbc522))
* add payload limits, admission control and metrics ([7543da1](https://github.com/mrwogu/factory-droid-openai/commit/7543da1ab8f11de59c1392b946f8df76c67a4794))
* add stop sequences, multiple choices and session continuity ([8fb67ad](https://github.com/mrwogu/factory-droid-openai/commit/8fb67ada00e3a3c9f1c140a3b1235ba207861335))
* add versioned OpenAPI contract and client compatibility ([1db001a](https://github.com/mrwogu/factory-droid-openai/commit/1db001a9eaf94704a8d2b45d5947f0673d24ad8a))
* emit OpenAI-compatible streaming usage chunks ([dc977f0](https://github.com/mrwogu/factory-droid-openai/commit/dc977f0c57c9cd3976f5f47fe5cff5c0b65b95bb))


### Performance Improvements

* scan only structural bytes for the JSON depth guard ([cb53a8f](https://github.com/mrwogu/factory-droid-openai/commit/cb53a8f1f7718bfdcf24deeb978db786273a8196))
