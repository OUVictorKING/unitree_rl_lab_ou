// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <unordered_map>

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            FSMStringMap.insert({id, fsm_name});
            specs_[id] = StateSpec{fsm_name, fsm_type};
            if (initial_state_id_ == 0 || fsm_name == "Passive")
                initial_state_id_ = id;
        }

        // Create only the initial state at startup. Other policies are loaded
        // lazily when their transition key is pressed, which keeps multi-policy
        // deploys from failing just because an unused checkpoint/motion asset
        // is stale.
        if (initial_state_id_ == 0)
            throw std::runtime_error("FSM: no initial state configured");
        add(create_state_(initial_state_id_));
    }

    void start() 
    {
        // Start From State_Passive
        currentState = states[0];
        currentState->enter();

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void stop()
    {
        if (stopped_)
            return;
        stopped_ = true;
        fsm_thread_.reset();
        if (currentState)
            currentState->exit();
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }
    
    ~CtrlFSM()
    {
        stop();
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    struct StateSpec
    {
        std::string name;
        std::string type;
    };

    const double dt = 0.001;
    int initial_state_id_ = 0;
    bool stopped_ = false;
    std::unordered_map<int, StateSpec> specs_;

    std::shared_ptr<BaseState> find_state_(int state_id)
    {
        for (auto &state : states)
        {
            if (state->isState(state_id))
                return state;
        }
        return nullptr;
    }

    std::shared_ptr<BaseState> create_state_(int state_id)
    {
        auto spec_it = specs_.find(state_id);
        if (spec_it == specs_.end())
            throw std::runtime_error("FSM: state id is not configured");

        const auto &spec = spec_it->second;
        auto fsm_class = getFsmMap().find("State_" + spec.type);
        if (fsm_class == getFsmMap().end())
            throw std::runtime_error("FSM: Unknown FSM type " + spec.type);
        return fsm_class->second(state_id, spec.name);
    }

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();
        
        // Check if need to change state
        int nextStateMode = 0;
        for(int i(0); i<currentState->registered_checks.size(); i++)
        {
            if(currentState->registered_checks[i].first())
            {
                nextStateMode = currentState->registered_checks[i].second;
                break;
            }
        }

        if(nextStateMode != 0 && !currentState->isState(nextStateMode))
        {
            auto state = find_state_(nextStateMode);
            if (!state)
            {
                try
                {
                    state = create_state_(nextStateMode);
                    add(state);
                }
                catch (const std::exception &e)
                {
                    spdlog::error("FSM: failed to create target state {}: {}", nextStateMode, e.what());
                    return;
                }
            }

            spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
            currentState->exit();
            currentState = state;
            currentState->enter();
        }
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};
