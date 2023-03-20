
import os.path
import numpy as np
from scipy.io import wavfile

def encode(pathToAudio,stringToEncode, phaseX):
    rate,audioData1 = wavfile.read(pathToAudio)
    stringToEncode = stringToEncode.ljust(100, '~')
    textLength = 8 * len(stringToEncode)

    chunkSize = int(2 * 2 ** np.ceil(np.log2(2 * textLength)))
    numberOfChunks = int(np.ceil(audioData1.shape[0] / chunkSize))
    audioData = audioData1.copy()

    # Tach am thanh thanh nhieu segment
    if len(audioData1.shape) == 1:
        audioData.resize(numberOfChunks * chunkSize, refcheck=False)
        audioData = audioData[np.newaxis]
    else:
        audioData.resize((numberOfChunks * chunkSize, audioData.shape[1]), refcheck=False)
        audioData = audioData.T

    chunks = audioData[0].reshape((numberOfChunks, chunkSize))

    #DFT voi tung segment
    chunks = np.fft.fft(chunks)
    magnitudes = np.abs(chunks)
    phases = np.angle(chunks)
    phaseDiff = np.diff(phases, axis=0)

    # chuyen thong diep sang nhi phan
    textInBinary = np.ravel([[int(y) for y in format(ord(x), "08b")] for x in stringToEncode])

    # Chuyen thong diep tu nhi phan sang pha 
    textInPi = textInBinary.copy()
    textInPi[textInPi == 0] = -1    
    textInPi = textInPi * -np.pi / 2
    midChunk = chunkSize // 2

    # Doi pha cua doan dau tien
    phases[phaseX, midChunk - textLength: midChunk] = textInPi
    phases[phaseX, midChunk + 1: midChunk + 1 + textLength] = -textInPi[::-1]

    # tinh lai ma tran pha 
    for i in range(phaseX - 1, -1, -1):
        phases[i] = phases[i + 1] - phaseDiff[i]

    for i in range(phaseX + 1, len(phases)):
        phases[i] = phases[i - 1] + phaseDiff[i - 1]
    

    # DFT nguoc 
    chunks = (magnitudes * np.exp(1j * phases))
    chunks = np.fft.ifft(chunks).real
    audioData[0] = chunks.ravel().astype(np.int16)    

    dir = os.path.dirname(pathToAudio)
    wavfile.write(dir + "/output.wav", rate, audioData.T)
    return dir + "/output.wav" 




def decode(audioLocation, phaseX):
    rate, audioData = wavfile.read(audioLocation)
    print(rate)
    textLength = 800
    blockLength = 2 * int(2 ** np.ceil(np.log2(2 * textLength)))
    blockMid = blockLength // 2
    print(blockLength, blockMid)

    # lay thong tin doan giau tin
    
    if len(audioData.shape) == 1:
        code = audioData[4096 * phaseX : 4096 * phaseX + blockLength]
    else:
        code = audioData[4096 * phaseX : 4096 * phaseX + blockLength, 0]
    print(code)

    # lay pha va chuyen thanh nhi phan
    codePhases = np.angle(np.fft.fft(code))[blockMid - textLength:blockMid]
    codeInBinary = (codePhases < 0).astype(np.int16)

    # chuyen doi thanh ky tu    
    codeInIntCode = codeInBinary.reshape((-1, 8)).dot(1 << np.arange(8 - 1, -1, -1))
    
    # ket hop ky tu thanh van ban
    return "".join(np.char.mod("%c", codeInIntCode)).replace("~", "")



# encode("/home/theanh/Desktop/demo/original_signal.wav", "theanh", 1)
print(decode("/home/theanh/Desktop/demo/output.wav", 1))      