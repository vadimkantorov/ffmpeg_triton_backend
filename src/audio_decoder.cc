#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>
#include <triton/core/tritonbackend.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

#include <vector>
#include <string>
#include <iostream>

#define RETURN_IF_ERROR(S)           \
  do {                               \
    TRITONSERVER_Error* error__ = (S); \
    if (error__ != nullptr) {        \
      return error__;                \
    }                                \
  } while (false)

namespace triton {
namespace backend {
namespace audio_decoder {

class AudioDecoder {
public:
    AudioDecoder() {
        avformat_network_init();
        av_log_set_level(AV_LOG_ERROR);
    }

    ~AudioDecoder() {
        avformat_network_deinit();
    }

    bool Decode(const uint8_t* input_data, size_t input_size, std::vector<uint8_t>& output_pcm) {
        AVFormatContext* format_ctx = avformat_alloc_context();
        if (!format_ctx) {
            std::cerr << "Failed to allocate format context\n";
            return false;
        }

        AVIOContext* avio_ctx = avio_alloc_context((unsigned char*)av_malloc(input_size), input_size, 0, nullptr, nullptr, nullptr, nullptr);
        if (!avio_ctx) {
            std::cerr << "Failed to allocate AVIO context\n";
            avformat_free_context(format_ctx);
            return false;
        }
        format_ctx->pb = avio_ctx;

        if (avformat_open_input(&format_ctx, nullptr, nullptr, nullptr) < 0) {
            std::cerr << "Could not open input format\n";
            avformat_free_context(format_ctx);
            return false;
        }

        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            std::cerr << "Failed to find stream info\n";
            avformat_close_input(&format_ctx);
            return false;
        }

        AVCodec* codec = nullptr;
        AVCodecContext* codec_ctx = nullptr;
        int stream_index = -1;
        
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                codec = avcodec_find_decoder(format_ctx->streams[i]->codecpar->codec_id);
                if (!codec) {
                    std::cerr << "Unsupported codec\n";
                    avformat_close_input(&format_ctx);
                    return false;
                }
                stream_index = i;
                break;
            }
        }

        if (stream_index == -1) {
            std::cerr << "No audio stream found\n";
            avformat_close_input(&format_ctx);
            return false;
        }

        codec_ctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(codec_ctx, format_ctx->streams[stream_index]->codecpar);
        
        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            std::cerr << "Failed to open codec\n";
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&format_ctx);
            return false;
        }

        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        SwrContext* swr_ctx = swr_alloc();
        av_opt_set_int(swr_ctx, "in_channel_layout", codec_ctx->channel_layout, 0);
        av_opt_set_int(swr_ctx, "out_channel_layout", AV_CH_LAYOUT_STEREO, 0);
        av_opt_set_int(swr_ctx, "in_sample_rate", codec_ctx->sample_rate, 0);
        av_opt_set_int(swr_ctx, "out_sample_rate", 44100, 0);
        av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);
        av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
        swr_init(swr_ctx);

        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == stream_index) {
                if (avcodec_send_packet(codec_ctx, packet) == 0) {
                    while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                        uint8_t* output;
                        int out_samples = swr_convert(swr_ctx, &output, frame->nb_samples, (const uint8_t**)frame->data, frame->nb_samples);
                        output_pcm.insert(output_pcm.end(), output, output + out_samples * 2);
                    }
                }
            }
            av_packet_unref(packet);
        }

        av_packet_free(&packet);
        av_frame_free(&frame);
        swr_free(&swr_ctx);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return true;
    }
};

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, uint32_t request_count) {

    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Input* input;
        TRITONBACKEND_RequestInput(requests[r], "INPUT_AUDIO", &input);

        const void* input_buffer;
        uint64_t input_byte_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_id;
        TRITONBACKEND_InputBuffer(input, 0, &input_buffer, &input_byte_size, &memory_type, &memory_id);

        std::vector<uint8_t> output_pcm;
        AudioDecoder decoder;
        decoder.Decode(static_cast<const uint8_t*>(input_buffer), input_byte_size, output_pcm);

        TRITONBACKEND_Output* output;
        TRITONBACKEND_RequestOutput(requests[r], "OUTPUT_PCM", &output);

        TRITONBACKEND_OutputBuffer(output, output_pcm.data(), output_pcm.size(), &memory_type, &memory_id);
        TRITONBACKEND_RequestRespond(requests[r], TRITONSERVER_ResponseComplete(requests[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL));
    }

    return nullptr;
}

}  // namespace audio_decoder
}  // namespace backend
}  // namespace triton


