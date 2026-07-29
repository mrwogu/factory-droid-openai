# Changelog

## [1.4.1](https://github.com/mrwogu/factory-droid-openai/compare/v1.4.0...v1.4.1) (2026-07-29)


### Bug Fixes

* **config:** reasoning effort override, teardown grace, dialect in protocol error ([#29](https://github.com/mrwogu/factory-droid-openai/issues/29)) ([dd2d105](https://github.com/mrwogu/factory-droid-openai/commit/dd2d10527d024c13f6cdbfa8b7c57defb94a3723))

## [1.4.0](https://github.com/mrwogu/factory-droid-openai/compare/v1.3.4...v1.4.0) (2026-07-29)


### Features

* **api:** serve retrieve-model and version probes ([#27](https://github.com/mrwogu/factory-droid-openai/issues/27)) ([77cdd13](https://github.com/mrwogu/factory-droid-openai/commit/77cdd13372a416e05590c06b841328aa382450f1))
* **protocol:** decode model tool-call dialects via registry ([#26](https://github.com/mrwogu/factory-droid-openai/issues/26)) ([4b45ed5](https://github.com/mrwogu/factory-droid-openai/commit/4b45ed5389cf3201f82eb6a59435cac5bddfa2df))
* **scripts:** show verify progress in vscode model generator ([8cb2736](https://github.com/mrwogu/factory-droid-openai/commit/8cb2736b611c072e5872870eb96dd4eb400dba2a))


### Bug Fixes

* **protocol:** harden model dialect decoding ([#28](https://github.com/mrwogu/factory-droid-openai/issues/28)) ([4197ac1](https://github.com/mrwogu/factory-droid-openai/commit/4197ac1f3326ef8963681475db1157c1822d879e))

## [1.3.4](https://github.com/mrwogu/factory-droid-openai/compare/v1.3.3...v1.3.4) (2026-07-28)


### Bug Fixes

* **attachments:** remove ReDoS in data URI parameter parsing ([#23](https://github.com/mrwogu/factory-droid-openai/issues/23)) ([60c0f94](https://github.com/mrwogu/factory-droid-openai/commit/60c0f949e7d3c2f3e09d12c50a4d09d2245b893e))

## [1.3.3](https://github.com/mrwogu/factory-droid-openai/compare/v1.3.2...v1.3.3) (2026-07-27)


### Bug Fixes

* **protocol:** repair malformed tool-call markers ([#21](https://github.com/mrwogu/factory-droid-openai/issues/21)) ([90e5bb2](https://github.com/mrwogu/factory-droid-openai/commit/90e5bb2cbe79dab86802c90cb5920d563e586ee7))

## [1.3.2](https://github.com/mrwogu/factory-droid-openai/compare/v1.3.1...v1.3.2) (2026-07-27)


### Bug Fixes

* **runner:** count only SIGKILL as a forced kill ([#19](https://github.com/mrwogu/factory-droid-openai/issues/19)) ([a0f183d](https://github.com/mrwogu/factory-droid-openai/commit/a0f183db4f854a285210cc60486894e17d50819d))

## [1.3.1](https://github.com/mrwogu/factory-droid-openai/compare/v1.3.0...v1.3.1) (2026-07-27)


### Bug Fixes

* **models:** withhold models an organization policy blocks ([#17](https://github.com/mrwogu/factory-droid-openai/issues/17)) ([b5e5100](https://github.com/mrwogu/factory-droid-openai/commit/b5e5100e7b29f3957294ea27c564abd296bb504c))


### Performance Improvements

* **pool:** retune warm sessions instead of cold starts ([#16](https://github.com/mrwogu/factory-droid-openai/issues/16)) ([3715616](https://github.com/mrwogu/factory-droid-openai/commit/37156164e80cff998231d94340ebcc875766dc54))

## [1.3.0](https://github.com/mrwogu/factory-droid-openai/compare/v1.2.1...v1.3.0) (2026-07-27)


### Features

* **logs:** add structured verbose logging with phase timings ([b442435](https://github.com/mrwogu/factory-droid-openai/commit/b4424359c7717b984b5acc059e47eb8a480b74a3))
* **perf:** serve requests from a warm Droid session pool ([0178b7e](https://github.com/mrwogu/factory-droid-openai/commit/0178b7ebce58ecead264812f0b81957d5919552e))


### Bug Fixes

* **ci:** build docker images without blocked third-party actions ([#15](https://github.com/mrwogu/factory-droid-openai/issues/15)) ([d75790f](https://github.com/mrwogu/factory-droid-openai/commit/d75790f87773b6d16c73327d7bf5d6f5d670059f))

## [1.2.1](https://github.com/mrwogu/factory-droid-openai/compare/v1.2.0...v1.2.1) (2026-07-27)


### Bug Fixes

* **ci:** pin valid docker action commit SHAs ([fd62b4e](https://github.com/mrwogu/factory-droid-openai/commit/fd62b4ef92bba95831120601830c2d839d8ee706))

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
