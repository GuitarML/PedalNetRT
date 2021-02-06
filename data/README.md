# Training data

1. Must be single channel, 44.1 kHz, FP32 wav data (not int16)
2. Wav files should be 3 - 4 minutes long, and contain a variety of
   chords, individual notes, and playing techniques to get a full spectrum
   of data for the model to "learn" from.
3. A buffer splitter was used with pedals to obtain a pure guitar signal
   and post effect signal.
4. Obtaining sample data from an amp can be done by splitting off the original
   signal, with the post amp signal coming from a microphone (I used a SM57).
   Keep in mind that this captures the dynamic response of the mic and cabinet.
   In the original research the sound was captured directly from within the amp
   circuit to have a "pure" amp signal.
5. When recording samples, try to maximize the volume levels without clipping.
   The levels you train the model at will be reproduced by the plugin. Also try
   to make the pre effect and post effect wav samples equal in volume levels.
   Even though the actual amp or effect may raise the level significantly, this isn't
   necessarily desirable in the end plugin.
