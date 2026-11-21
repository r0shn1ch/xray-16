package org.openxray.app;

import android.graphics.Rect;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;

/** Decodes game video to RGBA while honoring Android's flexible YUV plane strides. */
final class AndroidVideoPlayer {
    private static final String TAG = "OpenXRayVideo";
    private static final int MAX_FRAME_BYTES = 32 * 1024 * 1024;
    private static final int MAX_CODEC_STEPS_PER_UPDATE = 24;

    private MediaExtractor extractor;
    private MediaCodec codec;
    private MediaFormat outputFormat;
    private int width;
    private int height;
    private long durationUs;
    private long lastPtsUs;
    private long pendingPtsUs;
    private long startTimeMs;
    private long activeCycle;
    private boolean looped;
    private boolean playing;
    private boolean inputEos;
    private boolean outputEos;
    private byte[] pendingFrame;

    static AndroidVideoPlayer open(String path) {
        AndroidVideoPlayer player = new AndroidVideoPlayer();
        try {
            player.initialize(path);
            return player;
        } catch (Exception error) {
            Log.e(TAG, "Could not open game video", error);
            player.close();
            return null;
        }
    }

    int width() { return width; }
    int height() { return height; }
    boolean isPlaying() { return playing; }

    private void initialize(String path) throws IOException {
        extractor = new MediaExtractor();
        extractor.setDataSource(path);
        MediaFormat selected = null;
        int selectedTrack = -1;
        for (int track = 0; track < extractor.getTrackCount(); ++track) {
            MediaFormat format = extractor.getTrackFormat(track);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("video/")) {
                selected = format;
                selectedTrack = track;
                break;
            }
        }
        if (selected == null || selectedTrack < 0)
            throw new IOException("video container has no Android-decodable video track");

        width = visibleDimension(selected, MediaFormat.KEY_WIDTH,
                MediaFormat.KEY_CROP_LEFT, MediaFormat.KEY_CROP_RIGHT, 0);
        height = visibleDimension(selected, MediaFormat.KEY_HEIGHT,
                MediaFormat.KEY_CROP_TOP, MediaFormat.KEY_CROP_BOTTOM, 0);
        if (width <= 0 || height <= 0 || width > 8192 || height > 8192 ||
                (long) width * height * 4 > MAX_FRAME_BYTES)
            throw new IOException("Android MediaExtractor returned invalid video dimensions");
        durationUs = selected.containsKey(MediaFormat.KEY_DURATION)
                ? Math.max(0, selected.getLong(MediaFormat.KEY_DURATION)) : 0;

        extractor.selectTrack(selectedTrack);
        codec = MediaCodec.createDecoderByType(selected.getString(MediaFormat.KEY_MIME));
        codec.configure(selected, null, null, 0);
        codec.start();
        outputFormat = codec.getOutputFormat();
        refreshOutputDimensions(outputFormat);
    }

    private void refreshOutputDimensions(MediaFormat format) throws IOException {
        if (format == null)
            return;
        int outputWidth = visibleDimension(format, MediaFormat.KEY_WIDTH,
                MediaFormat.KEY_CROP_LEFT, MediaFormat.KEY_CROP_RIGHT, width);
        int outputHeight = visibleDimension(format, MediaFormat.KEY_HEIGHT,
                MediaFormat.KEY_CROP_TOP, MediaFormat.KEY_CROP_BOTTOM, height);
        if (outputWidth != width || outputHeight != height)
            throw new IOException("Android decoder changed video dimensions during playback");
    }

    void play(boolean shouldLoop, int timeMs) {
        if (codec == null || extractor == null)
            return;
        try {
            resetDecoder();
            startTimeMs = Integer.toUnsignedLong(timeMs);
            activeCycle = 0;
            looped = shouldLoop;
            playing = true;
        } catch (Exception error) {
            Log.e(TAG, "Could not restart Android video decoder", error);
            playing = false;
        }
    }

    void stop() { playing = false; }

    private void resetDecoder() {
        extractor.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC);
        codec.flush();
        inputEos = false;
        outputEos = false;
        pendingFrame = null;
        pendingPtsUs = 0;
        lastPtsUs = 0;
    }

    byte[] update(int gameTimeMs) {
        if (!playing || codec == null)
            return null;
        long elapsedMs = Integer.toUnsignedLong(gameTimeMs - (int) startTimeMs);
        long targetUs = elapsedMs * 1000;
        if (durationUs > 0 && targetUs >= durationUs) {
            if (!looped) {
                playing = false;
                return null;
            }
            long cycle = targetUs / durationUs;
            if (cycle != activeCycle) {
                try {
                    resetDecoder();
                    activeCycle = cycle;
                } catch (Exception error) {
                    Log.e(TAG, "Could not loop Android video decoder", error);
                    playing = false;
                    return null;
                }
            }
            targetUs %= durationUs;
        }

        byte[] decoded = null;
        if (pendingFrame != null) {
            if (pendingPtsUs > targetUs)
                return null;
            decoded = pendingFrame;
            pendingFrame = null;
        }

        for (int step = 0; step < MAX_CODEC_STEPS_PER_UPDATE; ++step) {
            feedInput();
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            final int outputIndex;
            try {
                outputIndex = codec.dequeueOutputBuffer(info, 0);
            } catch (IllegalStateException error) {
                Log.e(TAG, "Android video decoder output dequeue failed", error);
                playing = false;
                break;
            }
            if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                outputFormat = codec.getOutputFormat();
                try {
                    refreshOutputDimensions(outputFormat);
                } catch (IOException error) {
                    Log.e(TAG, error.getMessage(), error);
                    playing = false;
                    break;
                }
                continue;
            }
            if (outputIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED)
                continue;
            if (outputIndex < 0)
                break;

            boolean futureFrame = false;
            try {
                if (info.size > 0) {
                    byte[] frame = decodeOutput(outputIndex, info);
                    long ptsUs = Math.max(0, info.presentationTimeUs);
                    lastPtsUs = ptsUs;
                    if (ptsUs <= targetUs)
                        decoded = frame;
                    else {
                        pendingFrame = frame;
                        pendingPtsUs = ptsUs;
                        futureFrame = true;
                    }
                }
                if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0)
                    outputEos = true;
            } catch (RuntimeException | IOException error) {
                Log.e(TAG, "Android video frame conversion failed", error);
                playing = false;
            } finally {
                try { codec.releaseOutputBuffer(outputIndex, false); }
                catch (IllegalStateException error) {
                    Log.w(TAG, "Could not release Android video output buffer", error);
                }
            }
            if (!playing || futureFrame || outputEos)
                break;
        }

        if (outputEos && !looped && durationUs <= 0 && targetUs >= lastPtsUs)
            playing = false;
        else if (outputEos && looped && durationUs <= 0 && targetUs >= lastPtsUs) {
            try {
                resetDecoder();
                startTimeMs = Integer.toUnsignedLong(gameTimeMs);
                activeCycle = 0;
            } catch (Exception error) {
                Log.e(TAG, "Could not restart Android video at end of stream", error);
                playing = false;
            }
        }
        return decoded;
    }

    private void feedInput() {
        if (inputEos)
            return;
        final int inputIndex;
        try { inputIndex = codec.dequeueInputBuffer(0); }
        catch (IllegalStateException error) { playing = false; return; }
        if (inputIndex < 0)
            return;
        ByteBuffer input;
        try { input = codec.getInputBuffer(inputIndex); }
        catch (IllegalStateException error) { playing = false; return; }
        if (input == null) {
            playing = false;
            return;
        }
        input.clear();
        int sampleSize = extractor.readSampleData(input, 0);
        if (sampleSize < 0) {
            inputEos = true;
            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
            return;
        }
        long sampleTime = Math.max(0, extractor.getSampleTime());
        codec.queueInputBuffer(inputIndex, 0, sampleSize, sampleTime, 0);
        extractor.advance();
    }

    private byte[] decodeOutput(int outputIndex, MediaCodec.BufferInfo info) throws IOException {
        Image image = codec.getOutputImage(outputIndex);
        if (image != null) {
            try { return convertImage(image); }
            finally { image.close(); }
        }

        MediaFormat format = codec.getOutputFormat();
        int color = format.containsKey(MediaFormat.KEY_COLOR_FORMAT)
                ? format.getInteger(MediaFormat.KEY_COLOR_FORMAT) : 0;
        ByteBuffer buffer = codec.getOutputBuffer(outputIndex);
        if (buffer == null || info.size <= 0 || info.size > buffer.remaining())
            throw new IOException("Android decoder returned an invalid output buffer");
        if (color != 19 && color != 21 && color != 39)
            throw new IOException("Android decoder returned flexible YUV without accessible image planes");
        return convertPackedYuv(buffer.slice(), info.size, format, color);
    }

    private byte[] convertImage(Image image) throws IOException {
        if (image.getFormat() != android.graphics.ImageFormat.YUV_420_888)
            throw new IOException("Android decoder returned an unsupported image format");
        Image.Plane[] planes = image.getPlanes();
        if (planes.length != 3)
            throw new IOException("Android decoder returned an invalid flexible YUV plane count");
        Rect crop = image.getCropRect();
        int cropWidth = crop.width();
        int cropHeight = crop.height();
        if (cropWidth != width || cropHeight != height)
            throw new IOException("Android decoder returned an unsupported cropped video size");
        ByteBuffer y = planes[0].getBuffer().duplicate();
        ByteBuffer u = planes[1].getBuffer().duplicate();
        ByteBuffer v = planes[2].getBuffer().duplicate();
        int yBase = y.position();
        int uBase = u.position();
        int vBase = v.position();
        int[] rowStride = {planes[0].getRowStride(), planes[1].getRowStride(), planes[2].getRowStride()};
        int[] pixelStride = {planes[0].getPixelStride(), planes[1].getPixelStride(), planes[2].getPixelStride()};
        if (rowStride[0] <= 0 || rowStride[1] <= 0 || rowStride[2] <= 0 ||
                rowStride[0] > 65536 || rowStride[1] > 65536 || rowStride[2] > 65536 ||
                pixelStride[0] <= 0 || pixelStride[1] <= 0 || pixelStride[2] <= 0 ||
                pixelStride[0] > 16 || pixelStride[1] > 16 || pixelStride[2] > 16)
            throw new IOException("Android decoder returned invalid flexible YUV strides");
        MediaFormat format = codec.getOutputFormat();
        int standard = getFormatInt(format, MediaFormat.KEY_COLOR_STANDARD,
                MediaFormat.COLOR_STANDARD_BT601_NTSC);
        int range = getFormatInt(format, MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED);
        byte[] rgba = new byte[width * height * 4];
        for (int py = 0; py < height; ++py) {
            int sourceY = crop.top + py;
            for (int px = 0; px < width; ++px) {
                int sourceX = crop.left + px;
                int lumaIndex = sourceY * rowStride[0] + sourceX * pixelStride[0];
                int chromaX = sourceX / 2;
                int chromaY = sourceY / 2;
                int uIndex = chromaY * rowStride[1] + chromaX * pixelStride[1];
                int vIndex = chromaY * rowStride[2] + chromaX * pixelStride[2];
                if (yBase + lumaIndex >= y.limit() || uBase + uIndex >= u.limit() ||
                        vBase + vIndex >= v.limit())
                    throw new IOException("Android decoder returned truncated flexible YUV planes");
                writeRgba(rgba, (py * width + px) * 4, y.get(yBase + lumaIndex) & 0xff,
                        u.get(uBase + uIndex) & 0xff, v.get(vBase + vIndex) & 0xff,
                        standard, range);
            }
        }
        return rgba;
    }

    private byte[] convertPackedYuv(ByteBuffer source, int size, MediaFormat format, int color)
            throws IOException {
        int stride = format.containsKey(MediaFormat.KEY_STRIDE)
                ? format.getInteger(MediaFormat.KEY_STRIDE) : width;
        int sliceHeight = format.containsKey(MediaFormat.KEY_SLICE_HEIGHT)
                ? format.getInteger(MediaFormat.KEY_SLICE_HEIGHT) : height;
        int cropLeft = getFormatInt(format, MediaFormat.KEY_CROP_LEFT, 0);
        int cropTop = getFormatInt(format, MediaFormat.KEY_CROP_TOP, 0);
        if (sliceHeight <= 0) sliceHeight = height;
        if (cropLeft < 0 || cropLeft > 16384 || cropTop < 0 || cropTop > 16384 ||
                stride < cropLeft + width || stride > 16384 || sliceHeight < cropTop + height ||
                sliceHeight > 16384)
            throw new IOException("Android decoder returned invalid packed YUV strides");
        boolean planar = color == 19;
        int chromaStride = (stride + 1) / 2;
        int chromaRows = (sliceHeight + 1) / 2;
        long yBytes = (long) stride * sliceHeight;
        long chromaPlaneBytes = (long) chromaStride * chromaRows;
        long required = planar ? yBytes + chromaPlaneBytes * 2 : yBytes + (long) stride * chromaRows;
        if (required > size)
            throw new IOException("Android decoder returned a truncated packed YUV buffer");
        int standard = getFormatInt(format, MediaFormat.KEY_COLOR_STANDARD,
                MediaFormat.COLOR_STANDARD_BT601_NTSC);
        int range = getFormatInt(format, MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED);
        byte[] rgba = new byte[width * height * 4];
        for (int py = 0; py < height; ++py) {
            int sourceY = cropTop + py;
            for (int px = 0; px < width; ++px) {
                int sourceX = cropLeft + px;
                int yIndex = sourceY * stride + sourceX;
                int chromaIndex;
                int u;
                int v;
                if (planar) {
                    chromaIndex = (sourceY / 2) * chromaStride + sourceX / 2;
                    u = source.get((int) (yBytes + chromaIndex)) & 0xff;
                    v = source.get((int) (yBytes + chromaPlaneBytes + chromaIndex)) & 0xff;
                } else {
                    chromaIndex = (sourceY / 2) * stride + (sourceX / 2) * 2;
                    u = source.get((int) (yBytes + chromaIndex)) & 0xff;
                    v = source.get((int) (yBytes + chromaIndex + 1)) & 0xff;
                }
                writeRgba(rgba, (py * width + px) * 4, source.get(yIndex) & 0xff,
                        u, v, standard, range);
            }
        }
        return rgba;
    }

    private static void writeRgba(byte[] output, int index, int y, int u, int v,
            int standard, int range) {
        int d = u - 128;
        int e = v - 128;
        boolean bt709 = standard == MediaFormat.COLOR_STANDARD_BT709;
        int red, green, blue;
        if (range == MediaFormat.COLOR_RANGE_FULL) {
            int luma = y;
            red = 256 * luma + (bt709 ? 403 : 359) * e;
            green = 256 * luma - (bt709 ? 48 : 88) * d - (bt709 ? 120 : 183) * e;
            blue = 256 * luma + (bt709 ? 475 : 454) * d;
        } else {
            int luma = Math.max(0, y - 16);
            red = 298 * luma + (bt709 ? 459 : 409) * e;
            green = 298 * luma - (bt709 ? 55 : 100) * d - (bt709 ? 136 : 208) * e;
            blue = 298 * luma + (bt709 ? 541 : 516) * d;
        }
        output[index] = clamp((red + 128) >> 8);
        output[index + 1] = clamp((green + 128) >> 8);
        output[index + 2] = clamp((blue + 128) >> 8);
        output[index + 3] = (byte) 255;
    }

    private static byte clamp(int value) { return (byte) Math.max(0, Math.min(255, value)); }

    private static int getFormatInt(MediaFormat format, String key, int fallback) {
        return format.containsKey(key) ? format.getInteger(key) : fallback;
    }

    private static int visibleDimension(MediaFormat format, String dimensionKey,
            String cropStartKey, String cropEndKey, int fallback) {
        int dimension = getFormatInt(format, dimensionKey, fallback);
        if (dimension <= 0)
            return dimension;
        int start = getFormatInt(format, cropStartKey, 0);
        int end = getFormatInt(format, cropEndKey, dimension - 1);
        return start >= 0 && end >= start && end < dimension ? end - start + 1 : 0;
    }

    void close() {
        playing = false;
        pendingFrame = null;
        if (codec != null) {
            try { codec.stop(); } catch (IllegalStateException ignored) { }
            codec.release();
            codec = null;
        }
        if (extractor != null) {
            extractor.release();
            extractor = null;
        }
        outputFormat = null;
    }
}
