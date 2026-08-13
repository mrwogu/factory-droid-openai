# Changelog

## [1.6.11](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.10...v1.6.11) (2026-08-13)


### Bug Fixes

* **protocol:** tolerate one GLM value-close in bare calls ([40222b7](https://github.com/mrwogu/factory-droid-openai/commit/40222b7014aa27a6315ff7d33af60bbc0cc4f1f5)), closes [#65](https://github.com/mrwogu/factory-droid-openai/issues/65)
* **runner:** report newest Droid usage snapshot ([f07294f](https://github.com/mrwogu/factory-droid-openai/commit/f07294f5526e8cd3c66399ef79013e9e30a189dc)), closes [#70](https://github.com/mrwogu/factory-droid-openai/issues/70)

## [1.6.10](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.9...v1.6.10) (2026-08-13)


### Bug Fixes

* **protocol:** recover packed python-call tool calls ([af5cbde](https://github.com/mrwogu/factory-droid-openai/commit/af5cbde92ad1d5d084d92ebb510e517d0b8f314d))
* **protocol:** tighten python-call name spacing and residue ([ee26610](https://github.com/mrwogu/factory-droid-openai/commit/ee2661047039cbb07b05dfbb611ce1fa6af0a451))

## [1.6.9](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.8...v1.6.9) (2026-08-12)


### Bug Fixes

* **protocol:** bound packed bare-call repair work ([1c28ec4](https://github.com/mrwogu/factory-droid-openai/commit/1c28ec4fd5c362912c82d8d02a5a2aff957617d5))
* **protocol:** keep diagnostics on the tool-call limit ([a596b8f](https://github.com/mrwogu/factory-droid-openai/commit/a596b8f6f0e202e8f6e138bb3aeba7753f284e10))
* **protocol:** recover packed GLM bare calls ([a1ce6b1](https://github.com/mrwogu/factory-droid-openai/commit/a1ce6b1f9998682539f5a69fe958aa8759d66274))
* **telemetry:** count sequential tool-call limits ([eb6e22d](https://github.com/mrwogu/factory-droid-openai/commit/eb6e22d5a8db8591f2989244672bb3af0bdc986d))
* **telemetry:** count tool-call repairs by dialect ([9e41b32](https://github.com/mrwogu/factory-droid-openai/commit/9e41b32678e6f1266f289680f8892c43d02baa8b))

## [1.6.8](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.7...v1.6.8) (2026-08-11)


### Bug Fixes

* **runtime:** harden tool calls and sessions ([f5ff63d](https://github.com/mrwogu/factory-droid-openai/commit/f5ff63ddce763add34d8c465acda01e271e3519c))

## [1.6.7](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.6...v1.6.7) (2026-08-11)


### Bug Fixes

* **protocol:** recover mismatched GLM close token ([2c0e2d9](https://github.com/mrwogu/factory-droid-openai/commit/2c0e2d9d5233af7653c5184194ea780050fe70e6))
* **protocol:** recover partial closing markers ([9f5dcb4](https://github.com/mrwogu/factory-droid-openai/commit/9f5dcb4903ab582d099509b1521f259ad6a3fa4e))

## [1.6.6](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.5...v1.6.6) (2026-08-10)


### Bug Fixes

* **ci:** compare OpenAPI schemas semantically ([6b65f25](https://github.com/mrwogu/factory-droid-openai/commit/6b65f25c8e0b705283954c7736f2aa004af0943f))
* **ci:** verify the OpenAPI contract is regenerated ([aa7f27e](https://github.com/mrwogu/factory-droid-openai/commit/aa7f27efef40bebd766b31617a39883c6fa000b5))
* **docker:** support settings file mount ([b27d304](https://github.com/mrwogu/factory-droid-openai/commit/b27d3048151e324c3292047bba5ede23434bd97b))
* **docs:** correct notice, retune, and settings scope ([144dcbe](https://github.com/mrwogu/factory-droid-openai/commit/144dcbee601ed75cd406561c9644d45be4244150))
* **docs:** rewrap the retune and malformed-notice paragraphs ([852561a](https://github.com/mrwogu/factory-droid-openai/commit/852561afb8ecd4f43fb99e813a8728f7f9b375bc))
* **e2e:** cover warm-pool model switches ([61ff592](https://github.com/mrwogu/factory-droid-openai/commit/61ff5924e595896b7ba5b795b70fecc2a44352cc))
* **e2e:** harden model transition verdicts ([77f4aaf](https://github.com/mrwogu/factory-droid-openai/commit/77f4aaf640794bbbc627559096b40d69588dd77b))
* **e2e:** skip the opted-out continuation phase ([35e9975](https://github.com/mrwogu/factory-droid-openai/commit/35e9975ab701e4a940e60c2b41487a636f64ca5b))
* **openapi:** regenerate the spec for locked dependencies ([5b64856](https://github.com/mrwogu/factory-droid-openai/commit/5b648560cd0895f9527c737980cb3b83e2ba5d05))
* **pool:** require a fresh session for every model change ([aad67da](https://github.com/mrwogu/factory-droid-openai/commit/aad67da5f65208664cd5549662768eac25da4304))
* **protocol:** contain malformed Kimi tool calls ([51d39dd](https://github.com/mrwogu/factory-droid-openai/commit/51d39dd616211fa94951af531837fd34eabe9d9c))
* **protocol:** contain mangled tool calls without a call id ([7e4de3f](https://github.com/mrwogu/factory-droid-openai/commit/7e4de3f7d8ebe85619edb9af555ed2f8d0f840bf))
* **protocol:** keep mangled tool-call detection active after plain JSON ([105fd04](https://github.com/mrwogu/factory-droid-openai/commit/105fd0404901b72a75b924839eca7334d6b5daf2))
* **runner:** match fresh-session models by family ([01444bc](https://github.com/mrwogu/factory-droid-openai/commit/01444bc77f35e98c417364d1ad2574ad207eaf88))
* **sessions:** preserve continuation settings ([29d70ff](https://github.com/mrwogu/factory-droid-openai/commit/29d70ff5bff0e966fa19b9db9737153221256863))
* **sessions:** reject continuation model switches ([19fe33d](https://github.com/mrwogu/factory-droid-openai/commit/19fe33def51a586d9f290ad57bd3915bbcfc53b0))
* **sessions:** settle continuation mismatch before prompt work ([a7476f4](https://github.com/mrwogu/factory-droid-openai/commit/a7476f4aecc6d75f96284b866ca9d0860edcef04))

## [1.6.5](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.4...v1.6.5) (2026-08-08)


### Bug Fixes

* harden tool settlement and matrix verdicts ([472d5dc](https://github.com/mrwogu/factory-droid-openai/commit/472d5dc0a2e1846ec57574e9c3d7b2bf00e2dd0d))
* **protocol:** harden message JSON recovery ([f294026](https://github.com/mrwogu/factory-droid-openai/commit/f294026ce8abb516c3123a8611cfd4c5677beddb))
* **protocol:** reject echoed OpenAI transcripts ([381d1e0](https://github.com/mrwogu/factory-droid-openai/commit/381d1e03f8cbb90dc23c3a73952bf3acafdd65cb))
* retry native tool disable settlement ([0dce6f7](https://github.com/mrwogu/factory-droid-openai/commit/0dce6f79c3aaf3cc1f09f5908cc63e17e7a7c4f9))

## [1.6.4](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.3...v1.6.4) (2026-08-06)


### Bug Fixes

* **telemetry:** send unified collector envelope ([436f2d0](https://github.com/mrwogu/factory-droid-openai/commit/436f2d06e691d149ad844e826a083a4db631dbf1))

## [1.6.3](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.2...v1.6.3) (2026-08-02)


### Bug Fixes

* **protocol:** name the callable tools in the prompt ([d9ba819](https://github.com/mrwogu/factory-droid-openai/commit/d9ba8197cc5c4f4b6a167fa5c04720f6e9a2fa72))
* **runner:** map upstream model refusals to 404 ([4afb298](https://github.com/mrwogu/factory-droid-openai/commit/4afb298d35457cd2e1d3e42a337e2e84ac73035f))
* **runner:** tolerate the two tools Droid keeps callable ([46c5627](https://github.com/mrwogu/factory-droid-openai/commit/46c562769eb7cfc2d557982bbe8de8ef344fedb1))

## [1.6.2](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.1...v1.6.2) (2026-07-31)


### Bug Fixes

* **bridge:** prevent client tool-call hangs ([#45](https://github.com/mrwogu/factory-droid-openai/issues/45)) ([e98c5e6](https://github.com/mrwogu/factory-droid-openai/commit/e98c5e69bbea063fc7ba17290b5d89c6cdce3611))

## [1.6.1](https://github.com/mrwogu/factory-droid-openai/compare/v1.6.0...v1.6.1) (2026-07-31)


### Bug Fixes

* **telemetry:** categorize mid-stream timeouts ([#43](https://github.com/mrwogu/factory-droid-openai/issues/43)) ([df70d96](https://github.com/mrwogu/factory-droid-openai/commit/df70d96aa5be3e1eec68aea7c5e83f260b806077))

## [1.6.0](https://github.com/mrwogu/factory-droid-openai/compare/v1.5.0...v1.6.0) (2026-07-31)


### Features

* **protocol:** show a concrete tool-call example in the prompt ([#38](https://github.com/mrwogu/factory-droid-openai/issues/38)) ([bcc14ca](https://github.com/mrwogu/factory-droid-openai/commit/bcc14cada1276f453ee19595de22cd41a921141a))
* **telemetry:** add privacy-safe aggregate dimensions ([#42](https://github.com/mrwogu/factory-droid-openai/issues/42)) ([7e4221b](https://github.com/mrwogu/factory-droid-openai/commit/7e4221b02b76670bbc236144602b8a5e9c879b0b))


### Bug Fixes

* **api:** reject duplicate keys in the raw request body ([#39](https://github.com/mrwogu/factory-droid-openai/issues/39)) ([dd75d60](https://github.com/mrwogu/factory-droid-openai/commit/dd75d60428d843c9d9953da02969c3d805f09c42))
* **api:** stop+note on malformed tool calls instead of length ([#37](https://github.com/mrwogu/factory-droid-openai/issues/37)) ([54dd2d8](https://github.com/mrwogu/factory-droid-openai/commit/54dd2d80d53ce2132519b9406b41fb2c15d1e680))
* **protocol:** repair python-call and mangled arg_key tool calls ([#36](https://github.com/mrwogu/factory-droid-openai/issues/36)) ([5bb9159](https://github.com/mrwogu/factory-droid-openai/commit/5bb9159014a16ef64e67fd240cd02143a0b892c8))
* **runner:** report elapsed time in the timeout error message ([#40](https://github.com/mrwogu/factory-droid-openai/issues/40)) ([287d598](https://github.com/mrwogu/factory-droid-openai/commit/287d598c013ada5e9a470bf46e63582a81ca3066))

## [1.5.0](https://github.com/mrwogu/factory-droid-openai/compare/v1.4.1...v1.5.0) (2026-07-30)


### Features

* add anonymous aggregate telemetry ([#35](https://github.com/mrwogu/factory-droid-openai/issues/35)) ([b453124](https://github.com/mrwogu/factory-droid-openai/commit/b4531240b3df064e2fb3f9df89c7bf3373224941))
* **logging:** opt-in payload tracing to JSONL with redaction ([#31](https://github.com/mrwogu/factory-droid-openai/issues/31)) ([bf10ccb](https://github.com/mrwogu/factory-droid-openai/commit/bf10ccbb0c101654e453b3c4acadb21e7c6d706c))


### Bug Fixes

* **protocol:** degrade malformed tool-call JSON to length finish ([680f18e](https://github.com/mrwogu/factory-droid-openai/commit/680f18ee06bfcdc36790bc494ff44fb51b70b7a4))
* **protocol:** finish_reason=length on truncated tool calls ([#33](https://github.com/mrwogu/factory-droid-openai/issues/33)) ([372458d](https://github.com/mrwogu/factory-droid-openai/commit/372458db3074cc3b6f6e7a9ecdfdad544a38bb0e))

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
