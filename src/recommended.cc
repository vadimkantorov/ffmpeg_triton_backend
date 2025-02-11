// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
}

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace recommended {

//
// Backend that demonstrates the TRITONBACKEND API. This backend works
// for any model that has 1 input with any datatype and any shape and
// 1 output with the same shape and datatype as the input. The backend
// supports both batching and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

/////////////

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  // Delete the "global" state associated with the backend.
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Name of the input and output tensor
  const std::string& InputTensorName() const { return input_name_; }
  const std::string& OutputTensorName() const { return output_name_; }

  // Datatype of the input and output tensor
  TRITONSERVER_DataType TensorDataType() const { return datatype_; }

  // Shape of the input and output tensor as given in the model
  // configuration file. This shape will not include the batch
  // dimension (if the model has one).
  const std::vector<int64_t>& TensorNonBatchShape() const { return nb_shape_; }

  // Shape of the input and output tensor, including the batch
  // dimension (if the model has one). This method cannot be called
  // until the model is completely loaded and initialized, including
  // all instances of the model. In practice, this means that backend
  // should only call it in TRITONBACKEND_ModelInstanceExecute.
  TRITONSERVER_Error* TensorShape(std::vector<int64_t>& shape);

  // Validate that this model is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  std::string input_name_;
  std::string output_name_;

  TRITONSERVER_DataType datatype_;

  bool shape_initialized_;
  std::vector<int64_t> nb_shape_;
  std::vector<int64_t> shape_;
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), shape_initialized_(false)
{
  // Validate that the model's configuration matches what is supported
  // by this backend.
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::TensorShape(std::vector<int64_t>& shape)
{
  // This backend supports models that batch along the first dimension
  // and those that don't batch. For non-batch models the output shape
  // will be the shape from the model configuration. For batch models
  // the output shape will be the shape from the model configuration
  // prepended with [ -1 ] to represent the batch dimension. The
  // backend "responder" utility used below will set the appropriate
  // batch dimension value for each response. The shape needs to be
  // initialized lazily because the SupportsFirstDimBatching function
  // cannot be used until the model is completely loaded.
  if (!shape_initialized_) {
    bool supports_first_dim_batching;
    RETURN_IF_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
    if (supports_first_dim_batching) {
      shape_.push_back(-1);
    }

    shape_.insert(shape_.end(), nb_shape_.begin(), nb_shape_.end());
    shape_initialized_ = true;
  }

  shape = shape_;

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // If verbose logging is enabled, dump the model's configuration as
  // JSON into the console output.
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }

  // ModelConfig is the model configuration as a TritonJson
  // object. Use the TritonJson utilities to parse the JSON and
  // determine if the configuration is supported by this backend.
  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

  // The model must have exactly 1 input and 1 output.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must have 1 input"));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must have 1 output"));

  common::TritonJson::Value input, output;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

  // Record the input and output name in the model state.
  const char* input_name;
  size_t input_name_len;
  RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
  input_name_ = std::string(input_name);

  const char* output_name;
  size_t output_name_len;
  RETURN_IF_ERROR(
      output.MemberAsString("name", &output_name, &output_name_len));
  output_name_ = std::string(output_name);

  // Input and output must have same datatype
  std::string input_dtype, output_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
  RETURN_ERROR_IF_FALSE(
      input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected input and output datatype to match, got ") +
          input_dtype + " and " + output_dtype);
  datatype_ = ModelConfigDataTypeToTritonServerDataType(input_dtype);

  // Input and output must have same shape. Reshape is not supported
  // on either input or output so flag an error is the model
  // configuration uses it.
  triton::common::TritonJson::Value reshape;
  RETURN_ERROR_IF_TRUE(
      input.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("reshape not supported for input tensor"));
  RETURN_ERROR_IF_TRUE(
      output.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("reshape not supported for output tensor"));

  std::vector<int64_t> input_shape, output_shape;
  RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  RETURN_ERROR_IF_FALSE(
      input_shape == output_shape, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected input and output shape to match, got ") +
          backend::ShapeToString(input_shape) + " and " +
          backend::ShapeToString(output_shape));

  nb_shape_ = input_shape;

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
  }

  ModelState* model_state_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Collect various timestamps during the execution of this batch or
  // requests. These values are reported below before returning from
  // the function.

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // At this point, the backend takes ownership of 'requests', which
  // means that it is responsible for sending a response for every
  // request. From here, even if something goes wrong in processing,
  // the backend must return 'nullptr' from this function to indicate
  // success. Any errors and failures must be communicated via the
  // response objects.
  //
  // To simplify error handling, the backend utilities manage
  // 'responses' in a specific way and it is recommended that backends
  // follow this same pattern. When an error is detected in the
  // processing of a request, an appropriate error response is sent
  // and the corresponding TRITONBACKEND_Response object within
  // 'responses' is set to nullptr to indicate that the
  // request/response has already been handled and no further processing
  // should be performed for that request. Even if all responses fail,
  // the backend still allows execution to flow to the end of the
  // function so that statistics are correctly reported by the calls
  // to TRITONBACKEND_ModelInstanceReportStatistics and
  // TRITONBACKEND_ModelInstanceReportBatchStatistics.
  // RESPOND_AND_SET_NULL_IF_ERROR, and
  // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
  // backend_common.h that assist in this management of response
  // objects.

  // The backend could iterate over the 'requests' and process each
  // one separately. But for performance reasons it is usually
  // preferred to create batched input tensors that are processed
  // simultaneously. This is especially true for devices like GPUs
  // that are capable of exploiting the large amount parallelism
  // exposed by larger data sets.
  //
  // The backend utilities provide a "collector" to facilitate this
  // batching process. The 'collector's ProcessTensor function will
  // combine a tensor's value from each request in the batch into a
  // single contiguous buffer. The buffer can be provided by the
  // backend or 'collector' can create and manage it. In this backend,
  // there is not a specific buffer into which the batch should be
  // created, so use ProcessTensor arguments that cause collector to
  // manage it. ProcessTensor does NOT support TRITONSERVER_TYPE_BYTES
  // data type.

  BackendInputCollector collector(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      false /* pinned_enabled */, nullptr /* stream*/);

  // To instruct ProcessTensor to "gather" the entire batch of input
  // tensors into a single contiguous buffer in CPU memory, set the
  // "allowed input types" to be the CPU ones (see tritonserver.h in
  // the triton-inference-server/core repo for allowed memory types).
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
      {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

  const char* input_buffer;
  size_t input_buffer_byte_size;
  TRITONSERVER_MemoryType input_buffer_memory_type;
  int64_t input_buffer_memory_type_id;

  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      collector.ProcessTensor(
          model_state->InputTensorName().c_str(), nullptr /* existing_buffer */,
          0 /* existing_buffer_byte_size */, allowed_input_types, &input_buffer,
          &input_buffer_byte_size, &input_buffer_memory_type,
          &input_buffer_memory_type_id));

  // Finalize the collector. If 'true' is returned, 'input_buffer'
  // will not be valid until the backend synchronizes the CUDA
  // stream or event that was used when creating the collector. For
  // this backend, GPU is not supported and so no CUDA sync should
  // be needed; so if 'true' is returned simply log an error.
  const bool need_cuda_input_sync = collector.Finalize();
  if (need_cuda_input_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'recommended' backend: unexpected CUDA sync required by collector");
  }

  // 'input_buffer' contains the batched input tensor. The backend can
  // implement whatever logic is necessary to produce the output
  // tensor. This backend simply logs the input tensor value and then
  // returns the input tensor value in the output tensor so no actual
  // computation is needed.

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ": requests in batch " +
       std::to_string(request_count))
          .c_str());
  std::string tstr;
  IGNORE_ERROR(BufferAsTypedString(
      tstr, input_buffer, input_buffer_byte_size,
      model_state->TensorDataType()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("batched " + model_state->InputTensorName() + " value: ") +
       tstr)
          .c_str());

  const char* output_buffer = input_buffer;
  TRITONSERVER_MemoryType output_buffer_memory_type = input_buffer_memory_type;
  int64_t output_buffer_memory_type_id = input_buffer_memory_type_id;

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  bool supports_first_dim_batching;
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      model_state->SupportsFirstDimBatching(&supports_first_dim_batching));

  std::vector<int64_t> tensor_shape;
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count, model_state->TensorShape(tensor_shape));

  // Because the output tensor values are concatenated into a single
  // contiguous 'output_buffer', the backend must "scatter" them out
  // to the individual response output tensors.  The backend utilities
  // provide a "responder" to facilitate this scattering process.
  // BackendOutputResponder does NOT support TRITONSERVER_TYPE_BYTES
  // data type.

  // The 'responders's ProcessTensor function will copy the portion of
  // 'output_buffer' corresponding to each request's output into the
  // response for that request.

  BackendOutputResponder responder(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      supports_first_dim_batching, false /* pinned_enabled */,
      nullptr /* stream*/);

  responder.ProcessTensor(
      model_state->OutputTensorName().c_str(), model_state->TensorDataType(),
      tensor_shape, output_buffer, output_buffer_memory_type,
      output_buffer_memory_type_id);

  // Finalize the responder. If 'true' is returned, the output
  // tensors' data will not be valid until the backend synchronizes
  // the CUDA stream or event that was used when creating the
  // responder. For this backend, GPU is not supported and so no CUDA
  // sync should be needed; so if 'true' is returned simply log an
  // error.
  const bool need_cuda_output_sync = responder.Finalize();
  if (need_cuda_output_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'recommended' backend: unexpected CUDA sync required by responder");
  }

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
  // For batch statistics need to know the total batch size of the
  // requests. This is not necessarily just the number of requests,
  // because if the model supports batching then any request can be a
  // batched request itself.
  size_t total_batch_size = 0;
  if (!supports_first_dim_batching) {
    total_batch_size = request_count;
  } else {
    for (uint32_t r = 0; r < request_count; ++r) {
      auto& request = requests[r];
      TRITONBACKEND_Input* input = nullptr;
      LOG_IF_ERROR(
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
          "failed getting request input");
      if (input != nullptr) {
        const int64_t* shape = nullptr;
        LOG_IF_ERROR(
            TRITONBACKEND_InputProperties(
                input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
            "failed getting input properties");
        if (shape != nullptr) {
          total_batch_size += shape[0];
        }
      }
    }
  }
#else
  (void)exec_start_ns;
  (void)exec_end_ns;
  (void)compute_start_ns;
  (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

  // Report statistics for each request, and then release the request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];

#ifdef TRITON_ENABLE_STATS
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
#endif  // TRITON_ENABLE_STATS

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

#ifdef TRITON_ENABLE_STATS
  // Report batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::recommended

/*

// based on https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/decode_audio.c and https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/demuxing_decoding.c and https://github.com/FFmpeg/FFmpeg/blob/master/doc/examples/filtering_audio.c

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>

// https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h
#include "dlpack.h"

void deleter(struct DLManagedTensor* self)
{
	if(self->dl_tensor.data)
	{
		free(self->dl_tensor.data);
		self->dl_tensor.data = NULL;
	}

	if(self->dl_tensor.shape)
	{
		free(self->dl_tensor.shape);
		self->dl_tensor.shape = NULL;
	}
	
	if(self->dl_tensor.strides)
	{
		free(self->dl_tensor.strides);
		self->dl_tensor.strides = NULL;
	}
}

void __attribute__ ((constructor)) onload()
{
	//needed before ffmpeg 4.0, deprecated in ffmpeg 4.0
	av_register_all();
	avfilter_register_all();
}

struct DecodeAudio
{
	char error[128];
	char fmt[8];
	uint64_t sample_rate;
	uint64_t num_channels;
	uint64_t num_samples;
	uint64_t itemsize;
	double duration;
	DLManagedTensor data;
};

void process_output_frame(uint8_t** data, AVFrame* frame, int num_samples, int num_channels, uint64_t* data_len, int itemsize)
{
	if(num_channels == 1)
		data = memcpy(*data, frame->data, itemsize * frame->nb_samples) + itemsize * frame->nb_samples;
	else
	{
		for (int i = 0; i < num_samples; i++)
		{
			for (int c = 0; c < num_channels; c++)
			{
				if(*data_len >= itemsize)
				{
					*data = memcpy(*data, frame->data[c] + itemsize * i, itemsize) + itemsize;
					*data_len -= itemsize;
				}
			}
		}
	}
}

int decode_packet(AVCodecContext *av_ctx, AVFilterContext* buffersrc_ctx, AVFilterContext* buffersink_ctx, AVPacket *pkt, uint8_t** data, uint64_t* data_len, int itemsize)
{
	AVFrame *frame = av_frame_alloc();
	AVFrame *filt_frame = av_frame_alloc();

	int ret = avcodec_send_packet(av_ctx, pkt);

	int filtering = buffersrc_ctx != NULL && buffersink_ctx != NULL;
	while (ret >= 0)
	{
		ret = avcodec_receive_frame(av_ctx, frame);
		if (ret == 0)
		{
			if(filtering)
			{
				ret = av_buffersrc_add_frame_flags(buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
				if(ret < 0)
					goto end;
			}
			
			while (filtering)
			{
				ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
					break;
				if (ret < 0)
					goto end;
				process_output_frame(data, filt_frame, filt_frame->nb_samples, av_ctx->channels, data_len, itemsize);
				av_frame_unref(filt_frame);
			}
			
			if(!filtering)
			{
				process_output_frame(data, frame, frame->nb_samples, av_ctx->channels, data_len, itemsize);
			}
			//av_frame_unref(frame);
		}
	}

end:
	if (ret == AVERROR(EAGAIN))
		ret = 0;
	
	av_frame_free(&frame);
	av_frame_free(&filt_frame);
	return ret;
}

struct buffer_cursor
{
	uint8_t *base;
	size_t size;
    uint8_t *ptr;
    size_t left;
};

static int buffer_read(void *opaque, uint8_t *buf, int buf_size)
{
    struct buffer_cursor *cursor = (struct buffer_cursor *)opaque;
    buf_size = FFMIN(buf_size, cursor->left);

    if (!buf_size)
        return AVERROR_EOF;

    memcpy(buf, cursor->ptr, buf_size);
    cursor->ptr += buf_size;
    cursor->left -= buf_size;
    return buf_size;
}

static int64_t buffer_seek(void* opaque, int64_t offset, int whence)
{
    struct buffer_cursor *cursor = (struct buffer_cursor *)opaque;
	if(whence == AVSEEK_SIZE)
		return cursor->size;

	cursor->ptr = cursor->base + offset;
	cursor->left = cursor->size - offset;
	return offset;
}

size_t nbytes(struct DecodeAudio* audio)
{
	size_t itemsize = audio->data.dl_tensor.dtype.lanes * audio->data.dl_tensor.dtype.bits / 8;
	size_t size = 1;
	for(size_t i = 0; i < audio->data.dl_tensor.ndim; i++)
		size *= audio->data.dl_tensor.shape[i];
	return size * itemsize;
}

struct DecodeAudio decode_audio(const char* input_path, struct DecodeAudio input_options, struct DecodeAudio output_options, const char* filter_string, int probe, int verbose)
{
	av_log_set_level(verbose ? AV_LOG_DEBUG : AV_LOG_FATAL);
	
	clock_t tic = clock();

	struct DecodeAudio audio = { 0 };

	AVIOContext* io_ctx = NULL;
	AVFormatContext* fmt_ctx = avformat_alloc_context();
	AVCodecContext* dec_ctx = NULL;
	AVPacket* pkt = NULL;
	char filter_args[1024];
	AVFilterGraph *graph = NULL;
	AVFilterInOut *gis = avfilter_inout_alloc();
    AVFilterInOut *gos = avfilter_inout_alloc();
	AVFilterContext *buffersrc_ctx = NULL;
    AVFilterContext *buffersink_ctx = NULL;
	AVFilter *buffersrc  = avfilter_get_by_name("abuffer");
    AVFilter *buffersink = avfilter_get_by_name("abuffersink");
	assert(buffersrc != NULL && buffersink != NULL);
	uint8_t* avio_ctx_buffer = NULL;
    struct buffer_cursor cursor = { 0 };
	int buffer_multiple = 1;
	
	if(filter_string != NULL && strlen(filter_string) > 512)
	{
		strcpy(audio.error, "Too long filter string");
		goto end;
	}

	if(input_path == NULL)
	{
		size_t avio_ctx_buffer_size = 4096 * buffer_multiple;
    	avio_ctx_buffer = av_malloc(avio_ctx_buffer_size);
		assert(avio_ctx_buffer);
    	
		cursor.base = cursor.ptr  = (uint8_t*)input_options.data.dl_tensor.data;
    	cursor.size = cursor.left = nbytes(&input_options);
		io_ctx = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 0, &cursor, &buffer_read, NULL, &buffer_seek);
		if(!io_ctx)
		{
			strcpy(audio.error, "Cannot allocate IO context");
			goto end;
		}

		fmt_ctx->pb = io_ctx;
	}

	if(verbose) printf("decode_audio_BEFORE__: %.2f microsec\n", (float)(clock() - tic) * 1000000 / CLOCKS_PER_SEC);

	fmt_ctx->format_probesize = 2048;
	AVInputFormat* input_format = av_find_input_format("wav");
	if (avformat_open_input(&fmt_ctx, input_path, input_format, NULL) != 0)
	{
		strcpy(audio.error, "Cannot open file");
		goto end;
	}
	if(probe) return audio;
	fmt_ctx->streams[0]->probe_packets = 1;
	//fmt_ctx->streams[0]->probesize = 2048;
	if(verbose) printf("decode_audio_BEFORE: %.2f microsec\n", (float)(clock() - tic) * 1000000 / CLOCKS_PER_SEC);
	
	//if (avformat_find_stream_info(fmt_ctx, NULL) < 0)
	//{
	//	strcpy(audio.error, "Cannot open find stream information");
	//	goto end;
	//}
	if(verbose) printf("decode_audio_AFTER: %.2f microsec\n", (float)(clock() - tic) * 1000000 / CLOCKS_PER_SEC);
	

	int stream_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
	if (stream_index < 0)
	{
		strcpy(audio.error, "Cannot find audio stream");
		goto end;
	}
	AVStream *stream = fmt_ctx->streams[stream_index];
	//stream->codecpar->block_align = 4096 * buffer_multiple;

	AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
	if (!codec)
	{
		strcpy(audio.error, "Codec not found");
		goto end;
	}

	dec_ctx = avcodec_alloc_context3(codec);
	if (!dec_ctx)
	{
		strcpy(audio.error, "Cannot allocate audio codec context");
		goto end;
	}

	if (avcodec_parameters_to_context(dec_ctx, stream->codecpar) < 0)
	{
		strcpy(audio.error, "Failed to copy audio codec parameters to decoder context");
		goto end;
	}

	if (avcodec_open2(dec_ctx, codec, NULL) < 0)
	{
		strcpy(audio.error, "Cannot open codec");
		goto end;
	}

	enum AVSampleFormat sample_fmt = dec_ctx->sample_fmt;
	if (av_sample_fmt_is_planar(sample_fmt))
	{
		const char *packed = av_get_sample_fmt_name(sample_fmt);
		printf("Warning: the sample format the decoder produced is planar (%s). This example will output the first channel only.\n", packed ? packed : "?");
		sample_fmt = av_get_packed_sample_fmt(dec_ctx->sample_fmt);
	}
	static struct sample_fmt_entry {enum AVSampleFormat sample_fmt; const char *fmt_be, *fmt_le; DLDataType dtype;} supported_sample_fmt_entries[] =
	{
		{ AV_SAMPLE_FMT_U8,  "u8"   ,    "u8" , { kDLUInt  , 8 , 1 }},
		{ AV_SAMPLE_FMT_S16, "s16be", "s16le" , { kDLInt   , 16, 1 }},
		{ AV_SAMPLE_FMT_S32, "s32be", "s32le" , { kDLInt   , 32, 1 }},
		{ AV_SAMPLE_FMT_FLT, "f32be", "f32le" , { kDLFloat , 32, 1 }},
		{ AV_SAMPLE_FMT_DBL, "f64be", "f64le" , { kDLFloat , 64, 1 }},
	};
	
	double in_duration = stream->time_base.num * (int)stream->duration / stream->time_base.den;
	//double in_duration = fmt_ctx->duration / (float) AV_TIME_BASE; assert(in_duration > 0);
	double out_duration = in_duration;
	int in_sample_rate = dec_ctx->sample_rate;
	int out_sample_rate = output_options.sample_rate > 0 ? output_options.sample_rate : in_sample_rate;
	uint64_t out_num_samples  = out_duration * out_sample_rate;
	int out_num_channels = dec_ctx->channels;

	DLDataType in_dtype, out_dtype;
	enum AVSampleFormat in_sample_fmt = AV_SAMPLE_FMT_NONE, out_sample_fmt = AV_SAMPLE_FMT_NONE; 
	for (int k = 0; k < FF_ARRAY_ELEMS(supported_sample_fmt_entries); k++)
	{
		struct sample_fmt_entry* entry = &supported_sample_fmt_entries[k];
		
		if (sample_fmt == entry->sample_fmt)
		{
			in_dtype = entry->dtype;
			in_sample_fmt = entry->sample_fmt;
            strcpy(audio.fmt, AV_NE(entry->fmt_be, entry->fmt_le));
		}

		if (strcmp(output_options.fmt, entry->fmt_le) == 0 || strcmp(output_options.fmt, entry->fmt_be) == 0)
		{
			out_dtype = entry->dtype;
			out_sample_fmt = entry->sample_fmt;
		}
	}
	if (in_sample_fmt == AV_SAMPLE_FMT_NONE)
	{
		strcpy(audio.error, "Cannot deduce format");
		goto end;
	}
	if (out_sample_fmt == AV_SAMPLE_FMT_NONE)
	{
		out_sample_fmt = in_sample_fmt;
		out_dtype = in_dtype;
	}

	if (!dec_ctx->channel_layout)
		dec_ctx->channel_layout = av_get_default_channel_layout(dec_ctx->channels);
	uint64_t channel_layout = dec_ctx->channel_layout;

	audio.duration = out_duration;
	audio.sample_rate = out_sample_rate;
	audio.num_channels = out_num_channels;
	audio.num_samples = out_num_samples;
	audio.data.dl_tensor.ctx.device_type = kDLCPU;
	audio.data.dl_tensor.ndim = 2;
	audio.data.dl_tensor.dtype = out_dtype; 
	audio.data.dl_tensor.shape = malloc(audio.data.dl_tensor.ndim * sizeof(int64_t));
	audio.data.dl_tensor.shape[0] = audio.num_samples;
	audio.data.dl_tensor.shape[1] = audio.num_channels;
	audio.data.dl_tensor.strides = malloc(audio.data.dl_tensor.ndim * sizeof(int64_t));
	audio.data.dl_tensor.strides[0] = audio.data.dl_tensor.shape[1];
	audio.data.dl_tensor.strides[1] = 1;
	audio.itemsize = audio.data.dl_tensor.dtype.lanes * audio.data.dl_tensor.dtype.bits / 8;
	
	if(probe)
		goto end;
    
	bool need_filter = filter_string != NULL && strlen(filter_string) > 0;
	bool need_resample = out_sample_rate != in_sample_rate || out_sample_fmt != in_sample_fmt;
	if(need_filter || need_resample)
	{
		graph = avfilter_graph_alloc();
		if(!graph)
		{
			strcpy(audio.error, "Cannot allocate filter graph");
			goto end;
		}

		sprintf(filter_args, "sample_rate=%d:sample_fmt=%s:channel_layout=0x%"PRIx64":time_base=%d/%d", in_sample_rate, av_get_sample_fmt_name(in_sample_fmt), channel_layout, dec_ctx->time_base.num, dec_ctx->time_base.den);

		if (avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in", filter_args, NULL, graph) < 0)
		{
			strcpy(audio.error, "Cannot create buffer source");
			goto end;
		}
		int ret;
		if ((ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out", NULL, NULL, graph)) < 0)
		{
			strcpy(audio.error, "Cannot create buffer sink");
			goto end;
		}
		const enum AVSampleFormat out_sample_fmts[] = { out_sample_fmt, -1 };
		if (av_opt_set_int_list(buffersink_ctx, "sample_fmts", out_sample_fmts, -1, AV_OPT_SEARCH_CHILDREN) < 0)
		{
			strcpy(audio.error, "Cannot set output sample format");
			goto end;
		}
		const int64_t out_channel_layouts[] = { channel_layout , -1 };
		if (av_opt_set_int_list(buffersink_ctx, "channel_layouts", out_channel_layouts, -1, AV_OPT_SEARCH_CHILDREN) < 0) 
		{
			strcpy(audio.error, "Cannot set output channel layout");
			goto end;
		}
		const int out_sample_rates[] = { out_sample_rate, -1 };
		if (av_opt_set_int_list(buffersink_ctx, "sample_rates", out_sample_rates, -1, AV_OPT_SEARCH_CHILDREN) < 0)
		{
			strcpy(audio.error, "Cannot set output sample rate");
			goto end;
		}
		
		const char* out_sample_fmt_name = av_get_sample_fmt_name(out_sample_fmt);
		if(need_resample)
		{
			sprintf(filter_args, "%s%saresample=out_sample_rate=%d:out_sample_fmt=%s,aformat=sample_rates=%d:sample_fmts=%s:channel_layouts=0x%"PRIu64, need_filter ? filter_string : "", need_filter ? "," : "", out_sample_rate, out_sample_fmt_name, out_sample_rate, out_sample_fmt_name, channel_layout);
		}
		else
		{
			sprintf(filter_args, "%s%saformat=sample_rates=%d:sample_fmts=%s:channel_layouts=0x%"PRIu64, need_filter ? filter_string : "", need_filter ? "," : "", out_sample_rate, out_sample_fmt_name, channel_layout);
		}
		
		gis->name = av_strdup("out");
		gis->filter_ctx = buffersink_ctx;
		gis->pad_idx = 0;
		gis->next = NULL;

		gos->name = av_strdup("in");
		gos->filter_ctx = buffersrc_ctx;
		gos->pad_idx = 0;
		gos->next = NULL;

		if(avfilter_graph_parse_ptr(graph, filter_args, &gis, &gos, NULL) < 0)
		{
			strcpy(audio.error, "Cannot parse graph");
			goto end;
		}

		if(avfilter_graph_config(graph, NULL) < 0)
		{
			strcpy(audio.error, "Cannot configure graph.");
			goto end;
		}
	}

	uint64_t data_len = 0;
	if(output_options.data.dl_tensor.data)
	{
		data_len = nbytes(&output_options);
		audio.data.dl_tensor.data = output_options.data.dl_tensor.data;
	}
	else
	{
		audio.data.deleter = deleter;
		data_len = audio.num_samples * audio.num_channels * audio.itemsize;
		audio.data.dl_tensor.data = calloc(data_len, 1);
	}

	uint8_t* data_ptr = audio.data.dl_tensor.data;
	pkt = av_packet_alloc();
	while (av_read_frame(fmt_ctx, pkt) >= 0)
	{
		//if (pkt->stream_index == stream_index && decode_packet(dec_ctx, buffersrc_ctx, buffersink_ctx, pkt, &data_ptr, &data_len, audio.itemsize) < 0)
		//	break;
		//av_packet_unref(pkt);
	}
	return audio;

	pkt->data = NULL;
	pkt->size = 0;
	//decode_packet(dec_ctx, buffersrc_ctx, buffersink_ctx, pkt, &data_ptr, &data_len, audio.itemsize);

end:
	if(graph)
		avfilter_graph_free(&graph);
	if(dec_ctx)
		avcodec_free_context(&dec_ctx);
	if(fmt_ctx)
		avformat_close_input(&fmt_ctx);
	if(pkt)
		av_packet_free(&pkt);
	if(gis)
		avfilter_inout_free(&gis);
	if(gos)
		avfilter_inout_free(&gos);
    if(io_ctx)
		av_free(io_ctx);
	
	//fprintf(stderr, "Error occurred: %s\n", av_err2str(ret));
	return audio;
}

int main(int argc, char **argv)
{
	if (argc <= 2)
	{
		printf("Usage: %s <input file> <output file> <filter string>\n", argv[0]);
		return 1;
	}
	
	struct DecodeAudio input_options = { 0 }, output_options = { 0 };
	
	//struct DecodeAudio audio = decode_audio(argv[1], false, input_options, output_options, argc == 4 ? argv[3] : NULL);

	char buf[100000];
	int64_t read = fread(buf, 1, sizeof(buf), fopen(argv[1], "r"));
	input_options.data.dl_tensor.data = &buf;
	input_options.data.dl_tensor.ndim = 1;
	input_options.data.dl_tensor.shape = &read;
	input_options.data.dl_tensor.dtype.lanes = 1;
	input_options.data.dl_tensor.dtype.bits = 8;
	input_options.data.dl_tensor.dtype.code = kDLUInt;
	
	clock_t tic = clock();
	struct DecodeAudio audio = decode_audio(NULL, input_options, output_options, argc == 4 ? argv[3] : NULL, false, true);
	printf("decode_audio: %.2f microsec\n", (float)(clock() - tic) * 1000000 / CLOCKS_PER_SEC);
	
	//printf("Error: [%s]\n", audio.error);
	//printf("ffplay -i %s\n", argv[1]);
	//printf("ffplay -f %s -ac %d -ar %d -i %s # num samples: %d\n", audio.fmt, (int)audio.num_channels, (int)audio.sample_rate, argv[2], (int)audio.num_samples);
	//fwrite(audio.data.dl_tensor.data, audio.itemsize, audio.num_samples * audio.num_channels, fopen(argv[2], "wb"));
	return 0;
}

#ifdef __cplusplus
extern "C"
{
  #define __STDC_CONSTANT_MACROS
  #include <libavutil/log.h>
  #include <libavutil/opt.h>
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
}
#endif
#include <string>
#include <vector>

struct Status {
  int status;
  std::string error;
};

struct AudioProperties {
  Status status;
  std::string encoding;
  int sample_rate;
  int channels;
  float duration;
};

struct DecodeAudioResult {
  Status status;
  std::vector<float> samples;
};

struct DecodeAudioOptions {
  bool multiChannel;
};

AudioProperties get_properties(const std::string& path);
DecodeAudioResult decode_audio(const std::string& path, float start, float duration, DecodeAudioOptions options);

#include "audio-decode.h"
#include <emscripten/bind.h>
#include <limits>
#include <cmath>


// * Reads the samples from a frame and puts them into the destination vector.
// * Samples will be stored as floats in the range of -1 to 1.
// * If the frame has multiple channels, samples will be averaged across all channels.
 
template <typename SampleType>
void read_samples(AVFrame* frame, std::vector<float>& dest, bool is_planar, bool multiChannel) {
  // use a midpoint offset between min/max for unsigned integer types
  SampleType min_numeric = std::numeric_limits<SampleType>::min();
  SampleType max_numeric = std::numeric_limits<SampleType>::max();
  SampleType zero_sample = min_numeric == 0 ? max_numeric / 2 + 1 : 0;

  for (int i = 0; i < frame->nb_samples; i++) {
    float sample = 0.0f;
    for (int j = 0; j < frame->channels; j++) {
      float channelSample = is_planar 
        ? (
          static_cast<float>(reinterpret_cast<SampleType*>(frame->extended_data[j])[i] - zero_sample) /
          static_cast<float>(max_numeric - zero_sample)
        )
        : (
          static_cast<float>(reinterpret_cast<SampleType*>(frame->data[0])[i * frame->channels + j] - zero_sample) / 
          static_cast<float>(max_numeric - zero_sample)
        );

      if (multiChannel) {
        dest.push_back(channelSample);
      } else {
        sample += channelSample;
      }
    }
    if (!multiChannel) {
      sample /= frame->channels;
      dest.push_back(sample);
    }
  }
}

template <>
void read_samples<float>(AVFrame* frame, std::vector<float>& dest, bool is_planar, bool multiChannel) {
    for (int i = 0; i < frame->nb_samples; i++) {
      float sample = 0.0f;
      for (int j = 0; j < frame->channels; j++) {
        float channelSample = is_planar 
          ? reinterpret_cast<float*>(frame->extended_data[j])[i]
          : reinterpret_cast<float*>(frame->data[0])[i * frame->channels + j];

        if (multiChannel) {
          dest.push_back(channelSample);
        } else {
          sample += channelSample;
        }
      }
      if (!multiChannel) {
        sample /= frame->channels;
        dest.push_back(sample);
      }
    }
}

int read_samples(AVFrame* frame, AVSampleFormat format, std::vector<float>& dest, bool multiChannel) {
  bool is_planar = av_sample_fmt_is_planar(format);
  switch (format) {
    case AV_SAMPLE_FMT_U8:
    case AV_SAMPLE_FMT_U8P:
      read_samples<uint8_t>(frame, dest, is_planar, multiChannel);
      return 0;
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S16P:
      read_samples<int16_t>(frame, dest, is_planar, multiChannel);
      return 0;
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_S32P:
      read_samples<int32_t>(frame, dest, is_planar, multiChannel);
      return 0;
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP:
      read_samples<float>(frame, dest, is_planar, multiChannel);
      return 0;
    default:
      return -1;
  }
}

std::string get_error_str(int status) {
  char errbuf[AV_ERROR_MAX_STRING_SIZE];
  av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, status);
  return std::string(errbuf);
}

Status open_audio_stream(const std::string& path, AVFormatContext*& format, AVCodecContext*& codec, int& audio_stream_index) {
  Status status;
  format = avformat_alloc_context();
  if ((status.status = avformat_open_input(&format, path.c_str(), nullptr, nullptr)) != 0) {
    status.error = "avformat_open_input: " + get_error_str(status.status);
    return status;
  }
  if ((status.status = avformat_find_stream_info(format, nullptr)) < 0) {
    status.error = "avformat_find_stream_info: " + get_error_str(status.status);
    return status;
  }
  AVCodec* decoder;
  if ((audio_stream_index = av_find_best_stream(format, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, -1)) < 0) {
    status.status = audio_stream_index;
    status.error = "av_find_best_stream: Failed to locate audio stream";
    return status;
  }
  codec = avcodec_alloc_context3(decoder);
  if (!codec) {
    status.status = -1;
    status.error = "avcodec_alloc_context3: Failed to allocate decoder";
    return status;
  }
  if ((status.status = avcodec_parameters_to_context(codec, format->streams[audio_stream_index]->codecpar)) < 0) {
    status.error = "avcodec_parameters_to_context: " + get_error_str(status.status);
    return status;
  }
  if ((status.status = avcodec_open2(codec, decoder, nullptr)) < 0) {
    status.error = "avcodec_open2: " + get_error_str(status.status);
    return status;
  }

  return status;
}

void close_audio_stream(AVFormatContext* format, AVCodecContext* codec, AVFrame* frame, AVPacket* packet) {
  if (format) {
    avformat_close_input(&format);
  }
  if (codec) {
    avcodec_free_context(&codec);
  }
  if (packet) {
    av_packet_free(&packet);
  }
  if (frame) {
    av_frame_free(&frame);
  }
}

AudioProperties get_properties(const std::string& path) {
  av_log_set_level(AV_LOG_ERROR);

  Status status;
  AVFormatContext* format;
  AVCodecContext* codec;
  int audio_stream_index;

  status = open_audio_stream(path, format, codec, audio_stream_index);
  if (status.status < 0) {
    close_audio_stream(format, codec, nullptr, nullptr);
    return { status };
  }
  AudioProperties properties = {
    status,
    avcodec_get_name(codec->codec_id),
    codec->sample_rate,
    codec->channels,
    format->duration / static_cast<float>(AV_TIME_BASE)
  };

  close_audio_stream(format, codec, nullptr, nullptr);
  return properties;
}

DecodeAudioResult decode_audio(const std::string& path, float start = 0, float duration = -1, DecodeAudioOptions options = {}) {
  av_log_set_level(AV_LOG_ERROR);

  Status status;
  AVFormatContext* format;
  AVCodecContext* codec;
  int audio_stream_index;

  status = open_audio_stream(path, format, codec, audio_stream_index);
  if (status.status < 0) {
    close_audio_stream(format, codec, nullptr, nullptr);
    // check if vector is undefined/null in js
    return { status };
  }

  // seek to start timestamp
  AVStream* stream = format->streams[audio_stream_index];
  int64_t start_timestamp = av_rescale(start, stream->time_base.den, stream->time_base.num);
  int64_t max_timestamp = av_rescale(format->duration / static_cast<float>(AV_TIME_BASE), stream->time_base.den, stream->time_base.num);
  if ((status.status = av_seek_frame(format, audio_stream_index, std::min(start_timestamp, max_timestamp), AVSEEK_FLAG_ANY)) < 0) {
    close_audio_stream(format, codec, nullptr, nullptr);
    status.error = "av_seek_frame: " + get_error_str(status.status) + ". timestamp: " + std::to_string(start);
    return { status };
  }

  AVPacket* packet = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  if (!packet || !frame) {
    close_audio_stream(format, codec, frame, packet);
    status.status = -1;
    status.error = "av_packet_alloc/av_frame_alloc: Failed to allocate decoder frame";
    return { status };
  }

  // decode loop
  std::vector<float> samples;
  int samples_to_decode = std::ceil(duration * codec->sample_rate);
  while ((status.status = av_read_frame(format, packet)) >= 0) {
    if (packet->stream_index == audio_stream_index) {
      // send compressed packet to decoder
      status.status = avcodec_send_packet(codec, packet);
      if (status.status == AVERROR(EAGAIN) || status.status == AVERROR_EOF) {
        continue;
      } else if (status.status < 0) {
        close_audio_stream(format, codec, frame, packet);
        status.error = "avcodec_send_packet: " + get_error_str(status.status);
        return { status };
      }

      // receive uncompressed frame from decoder
      while ((status.status = avcodec_receive_frame(codec, frame)) >= 0) {
        if (status.status == AVERROR(EAGAIN) || status.status == AVERROR_EOF) {
          break;
        } else if (status.status < 0) {
          close_audio_stream(format, codec, frame, packet);
          status.error = "avcodec_receive_frame: " + get_error_str(status.status);
          return { status };
        }

        // read samples from frame into result
        read_samples(frame, codec->sample_fmt, samples, options.multiChannel);
        av_frame_unref(frame);
      }

      av_packet_unref(packet);

      // stop decoding if we have enough samples
      if (samples_to_decode >= 0 && static_cast<int>(samples.size()) >= samples_to_decode) {
        break;
      }
    }
  }

  // cleanup
  close_audio_stream(format, codec, frame, packet);

  // success
  status.status = 0;
  return { status, samples };
}

EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::value_object<Status>("Status")
    .field("status", &Status::status)
    .field("error", &Status::error);
  emscripten::value_object<AudioProperties>("AudioProperties")
    .field("status", &AudioProperties::status)
    .field("encoding", &AudioProperties::encoding)
    .field("sampleRate", &AudioProperties::sample_rate)
    .field("channelCount", &AudioProperties::channels)
    .field("duration", &AudioProperties::duration);
  emscripten::value_object<DecodeAudioResult>("DecodeAudioResult")
    .field("status", &DecodeAudioResult::status)
    .field("samples", &DecodeAudioResult::samples);
  emscripten::value_object<DecodeAudioOptions>("DecodeAudioOptions")
    .field("multiChannel", &DecodeAudioOptions::multiChannel);
  emscripten::function("getProperties", &get_properties);
  emscripten::function("decodeAudio", &decode_audio);
  emscripten::register_vector<float>("vector<float>");
}
*/

