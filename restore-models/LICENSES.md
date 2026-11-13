# Model licences

These ONNX files are conversions of published models. The weights are unchanged
apart from being stored as 16-bit floats.

- **Sidon v0.1**: Wataru Nakata, Yuki Saito (SARU Lab), https://github.com/sarulab-speech/Sidon (MIT).
  Weights: https://huggingface.co/sarulab-speech/sidon-v0.1 (MIT), fine-tuned from
  w2v-BERT 2.0 (Meta, MIT).
- **PASE**: Cisco Systems, https://github.com/cisco-open/pase (Apache-2.0; it includes
  WavLM, Microsoft, MIT, and Vocos/WavTokenizer code, MIT).
  Weights: https://huggingface.co/cisco-ai/pase (Apache-2.0).
- **FlowSE**: Ziqian Wang et al., https://github.com/Honee-W/FlowSE (the repository states
  no licence). Weights: https://huggingface.co/flowse/wenetspeech4tts_Premium.pt.tar
  (Apache-2.0). The vocabulary file is FlowSE's Emilia_ZH_EN_pinyin/vocab.txt.
- **Vocos** (FlowSE's vocoder): Charactr, https://github.com/gemelo-ai/vocos (MIT).
  Weights: https://huggingface.co/charactr/vocos-mel-24khz (MIT).
