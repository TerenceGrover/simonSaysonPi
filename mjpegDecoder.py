def mjpeg_frames_from_pipe(pipe):
    buf = bytearray()
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        buf.extend(chunk)

        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                if len(buf) > 1_000_000:
                    del buf[:-100_000]
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                if start > 0:
                    del buf[:start]
                break

            jpg = bytes(buf[start:end + 2])
            del buf[:end + 2]
            yield jpg