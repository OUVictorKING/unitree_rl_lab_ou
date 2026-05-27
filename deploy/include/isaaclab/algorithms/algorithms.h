// // Copyright (c) 2025, Unitree Robotics Co., Ltd.
// // All rights reserved.

// #pragma once

// #include "onnxruntime_cxx_api.h"
// #include <iostream>
// #include <mutex>

// namespace isaaclab
// {

// class Algorithms
// {
// public:
//     virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

//     std::vector<float> get_action()
//     {
//         std::lock_guard<std::mutex> lock(act_mtx_);
//         return action;
//     }
    
//     std::vector<float> action;
// protected:
//     std::mutex act_mtx_;
// };

// class OrtRunner : public Algorithms
// {
// public:
//     OrtRunner(std::string model_path)
//     {
//         // Init Model
//         env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
//         session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

//         session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

//         for (size_t i = 0; i < session->GetInputCount(); ++i) {
//             Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
//             input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
//             auto input_name = session->GetInputNameAllocated(i, allocator);
//             input_names.push_back(input_name.release());
//         }

//         for (const auto& shape : input_shapes) {
//             size_t size = 1;
//             for (const auto& dim : shape) {
//                 size *= dim;
//             }
//             input_sizes.push_back(size);
//         }

//         // Get output shape
//         Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
//         output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
//         auto output_name = session->GetOutputNameAllocated(0, allocator);
//         output_names.push_back(output_name.release());

//         action.resize(output_shape[1]);
//     }

//     std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
//     {
//         auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

//         // make sure all input names are in obs
//         for (const auto& name : input_names) {
//             if (obs.find(name) == obs.end()) {
//                 throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
//             }
//         }

//         // Create input tensors
//         std::vector<Ort::Value> input_tensors;
//         for(int i(0); i<input_names.size(); ++i)
//         {
//             const std::string name_str(input_names[i]);
//             auto& input_data = obs.at(name_str);
//             auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
//             input_tensors.push_back(std::move(input_tensor));
//         }

//         // Run the model
//         auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

//         // Copy output data
//         auto floatarr = output_tensor.front().GetTensorMutableData<float>();
//         std::lock_guard<std::mutex> lock(act_mtx_);
//         std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
//         return action;
//     }

// private:
//     Ort::Env env;
//     Ort::SessionOptions session_options;
//     std::unique_ptr<Ort::Session> session;
//     Ort::AllocatorWithDefaultOptions allocator;

//     std::vector<const char*> input_names;
//     std::vector<const char*> output_names;

//     std::vector<std::vector<int64_t>> input_shapes;
//     std::vector<int64_t> input_sizes;
//     std::vector<int64_t> output_shape;
// };
// };

// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

namespace isaaclab
{

    class Algorithms
    {
    public:
        virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

        std::vector<float> get_action()
        {
            std::lock_guard<std::mutex> lock(act_mtx_);
            return action;
        }

        std::vector<float> action;

    protected:
        std::mutex act_mtx_;
    };

    class OrtRunner : public Algorithms
    {
    public:
        OrtRunner(std::string model_path)
        {
            env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
            session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

            std::cerr << "[ORT] loading model: " << model_path << std::endl;
            std::cerr << "[ORT] input_count=" << session->GetInputCount()
                      << ", output_count=" << session->GetOutputCount() << std::endl;

            // Inputs
            for (size_t i = 0; i < session->GetInputCount(); ++i)
            {
                Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
                auto raw_shape = input_type.GetTensorTypeAndShapeInfo().GetShape();
                input_shapes.push_back(raw_shape);

                auto input_name = session->GetInputNameAllocated(i, allocator);
                input_names.push_back(input_name.release());

                std::cerr << "[ORT] raw input[" << i << "] name=" << input_names.back() << " shape=";
                for (auto dim : raw_shape)
                    std::cerr << dim << " ";
                std::cerr << std::endl;
            }

            // Output
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
            output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
            auto output_name = session->GetOutputNameAllocated(0, allocator);
            output_names.push_back(output_name.release());

            std::cerr << "[ORT] raw output[0] name=" << output_names.back() << " shape=";
            for (auto dim : output_shape)
                std::cerr << dim << " ";
            std::cerr << std::endl;
        }

        std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
        {
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

            for (const auto &name : input_names)
            {
                if (obs.find(name) == obs.end())
                {
                    throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
                }
            }

            std::vector<Ort::Value> input_tensors;
            input_tensors.reserve(input_names.size());

            for (size_t i = 0; i < input_names.size(); ++i)
            {
                const std::string name_str(input_names[i]);
                auto &input_data = obs.at(name_str);

                auto used_shape = sanitize_input_shape(input_shapes[i], input_data.size());

                size_t expected_size = get_num_elements(used_shape);
                if (expected_size != input_data.size())
                {
                    std::cerr << "[ORT][ERROR] input[" << i << "] name=" << name_str
                              << " expected_size=" << expected_size
                              << " but got input_data.size()=" << input_data.size()
                              << " | shape=";
                    for (auto dim : used_shape)
                        std::cerr << dim << " ";
                    std::cerr << std::endl;

                    throw std::runtime_error("OrtRunner input size mismatch for input: " + name_str);
                }

                // std::cerr << "[ORT] used input[" << i << "] name=" << name_str << " shape=";
                // for (auto dim : used_shape)
                //     std::cerr << dim << " ";
                // std::cerr << " | flat_size=" << input_data.size() << std::endl;

                auto input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    input_data.data(),
                    input_data.size(),
                    used_shape.data(),
                    used_shape.size());

                input_tensors.push_back(std::move(input_tensor));
            }

            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                input_tensors.data(),
                input_tensors.size(),
                output_names.data(),
                1);

            auto &out0 = output_tensors.front();
            auto out_info = out0.GetTensorTypeAndShapeInfo();
            auto real_output_shape = out_info.GetShape();
            size_t real_output_size = out_info.GetElementCount();

            // std::cerr << "[ORT] real output shape=";
            // for (auto dim : real_output_shape)
            //     std::cerr << dim << " ";
            // std::cerr << " | real_output_size=" << real_output_size << std::endl;

            float *floatarr = out0.GetTensorMutableData<float>();

            {
                std::lock_guard<std::mutex> lock(act_mtx_);
                action.resize(real_output_size);
                std::memcpy(action.data(), floatarr, real_output_size * sizeof(float));
            }

            return action;
        }

    private:
        static std::vector<int64_t> sanitize_input_shape(
            const std::vector<int64_t> &raw_shape,
            size_t flat_input_size)
        {
            if (raw_shape.empty())
            {
                throw std::runtime_error("OrtRunner: input shape is empty");
            }

            std::vector<int64_t> shape = raw_shape;

            // 常见情况:
            //   [-1, obs_dim] -> [1, obs_dim]
            //   [obs_dim]     -> [obs_dim]
            //   [-1]          -> [flat_input_size] 或 [1] 不合理，这里直接按总长度替
            if (shape.size() == 1)
            {
                if (shape[0] < 0)
                {
                    shape[0] = static_cast<int64_t>(flat_input_size);
                }
            }
            else
            {
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    if (shape[i] < 0)
                    {
                        if (i == 0)
                        {
                            shape[i] = 1; // batch=1
                        }
                    }
                }

                // 如果最后一维还是动态，且前面维度已经能确定为 batch=1，
                // 则直接用 flat_input_size 补最后一维
                if (shape.back() < 0)
                {
                    shape.back() = static_cast<int64_t>(flat_input_size);
                }
            }

            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (shape[i] <= 0)
                {
                    throw std::runtime_error(
                        "OrtRunner: unresolved or invalid input dim at axis " + std::to_string(i));
                }
            }

            return shape;
        }

        static size_t get_num_elements(const std::vector<int64_t> &shape)
        {
            size_t n = 1;
            for (auto dim : shape)
            {
                if (dim <= 0)
                {
                    throw std::runtime_error("OrtRunner: invalid dim when computing tensor size");
                }
                n *= static_cast<size_t>(dim);
            }
            return n;
        }

    private:
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;
        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<const char *> input_names;
        std::vector<const char *> output_names;

        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<int64_t> output_shape;
    };

} // namespace isaaclab