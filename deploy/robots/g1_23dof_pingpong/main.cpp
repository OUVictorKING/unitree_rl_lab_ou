#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_Pingpong.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <thread>

namespace
{
volatile std::sig_atomic_t g_shutdown_requested = 0;
volatile std::sig_atomic_t g_signal_count = 0;

void handle_signal(int sig)
{
    g_shutdown_requested = 1;
    g_signal_count += 1;
    if (g_signal_count >= 2)
        std::_Exit(128 + sig);
}
}

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state(const std::string &network)
{
    const bool local_sim = (network == "lo" || network == "lo0");
    if(!local_sim)
    {
        auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
        usleep(0.2 * 1e6);
        if(!lowcmd_sub->isTimeout())
        {
            spdlog::critical("The other process is using the lowcmd channel, please close it first.");
            std::_Exit(EXIT_FAILURE);
        }
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Pingpong Controller \n";

    // Unitree DDS Config
    const std::string network = vm["network"].as<std::string>();
    const bool local_sim = (network == "lo" || network == "lo0");
    State_Pingpong::set_use_local_sim_time(local_sim);
    unitree::robot::ChannelFactory::Instance()->Init(0, network);

    init_fsm_state(network);

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    while (!g_shutdown_requested)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    spdlog::info("Shutdown requested, stopping FSM. Press Ctrl+C again to force exit.");
    fsm->stop();
    if (rclcpp::ok())
        rclcpp::shutdown();
    
    return 0;
}
