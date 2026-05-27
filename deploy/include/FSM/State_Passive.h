// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

class State_Passive : public FSMState
{
public:
    State_Passive(int state, std::string state_string = "Passive") 
    : FSMState(state, state_string) 
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if(motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for(int i(0); i<values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }
    } 

    void enter()
    {
        // set gain
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kd.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0;
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
    }

    // void run()
    // {
    //     for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
    //     {
    //         lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
    //     }
    // }


    // 修改run，一直print，在passive状态下对关节顺序进行测试验证
    // 期望：
    void run()
    {
        // 先保持原本 Passive 行为：跟随当前关节角，尽量不主动拉动机器人
        for (int i = 0; i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }

        // 23DoF action 顺序对应的 sdk 槽位
        static const int kJointIdsMap[23] = {
            0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26};

        static const char *kJointNames[23] = {
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint"};

        // 每 500 个周期打印一次；FSM 线程是 1kHz，这里大约每 0.5 秒打印一次
        // static int print_count = 0;
        // if (print_count++ % 500 == 0)
        // {
        //     std::cerr << "\n[PASSIVE READBACK CHECK]\n";

        //     for (int i = 0; i < 23; ++i)
        //     {
        //         int sdk_id = kJointIdsMap[i];
        //         const auto &ms = lowstate->msg_.motor_state()[sdk_id];

        //         std::cerr << "idx=" << i
        //                   << " joint=" << kJointNames[i]
        //                   << " sdk=" << sdk_id
        //                   << " q=" << ms.q()
        //                   << " dq=" << ms.dq()
        //                   << " tau_est=" << ms.tau_est()
        //                   << std::endl;
        //     }

        //     std::cerr << "joystick: "
        //               << "lx=" << lowstate->joystick.lx()
        //               << " ly=" << lowstate->joystick.ly()
        //               << " rx=" << lowstate->joystick.rx()
        //               << " ry=" << lowstate->joystick.ry()
        //               << std::endl;
        // }
    }
};

REGISTER_FSM(State_Passive)
