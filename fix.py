import av, os, soundfile as sf, numpy as np
def fix_file(path):
    print('Fixing', path)
    try:
        container = av.open(path)
        audio = []
        rate = container.streams.audio[0].rate
        for frame in container.decode(audio=0):
            audio.append(frame.to_ndarray())
        if audio:
            audio_data = np.concatenate(audio, axis=1).T
            sf.write(path, audio_data, rate)
            print('Converted', path)
    except Exception as e:
        print('Error:', path, e)
for d in ['data/raw/clean_speech', 'data/raw/noise']:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith('.wav'):
                fix_file(os.path.join(d, f))
