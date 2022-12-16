import airsim

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import argparse

#几何PD控制
def pd(opt, p_des, v_des, a_des, p_now, v_now, R_now, omega_now):
    yaw_des = opt.psid
    kp = opt.kp
    kv = opt.kv
    kR = opt.kR
    komega = opt.komega
    m = 1
    g = 9.81  
    #位置误差和速度误差
    e_p = p_now - p_des
    e_v = v_now - v_des
    e3 = np.array([[0], [0], [1]])
    # 求合力 f
    acc = -kp*e_p -kv*e_v - m*g*e3 + m*a_des   # 3x1
    f = -acc[2,0]
    # 求期望的旋转矩阵 R_des
    proj_xb = np.array([math.cos(yaw_des), math.sin(yaw_des), 0])
    acc = acc.reshape(3)
    z_b = - acc / np.linalg.norm(acc)
    y_b = np.cross(z_b, proj_xb)
    y_b = y_b / np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    x_b = x_b / np.linalg.norm(x_b)
    R_des = np.hstack([np.hstack([x_b.reshape([3, 1]), y_b.reshape([3, 1])]), z_b.reshape([3, 1])])
    # 获得姿态角误差
    e_R_tem = np.dot(R_des.T, R_now) - np.dot(R_now.T, R_des)/2
    e_R = np.array([[e_R_tem[2, 1]], [e_R_tem[0, 2]], [e_R_tem[1, 0]]])
    #求 力矩M
    M = -kR * e_R - komega * omega_now
    return f, M

#超前控制
def lead_control(opt, xk, u_last):
    g = opt.g
    m = opt.m
    Ix = opt.Ix
    Iy = opt.Iy
    Iz = opt.Iz
    xd = opt.xd
    yd = opt.yd
    zd = -opt.zd
    psid = opt.psid
    
    x1 = xk[0]  # x1 = phi
    x2 = xk[1]  # x2 = theta
    x3 = xk[2]  # x3 = psi
    x4 = xk[3]  # x4 = p
    x5 = xk[4]  # x5 = q
    x6 = xk[5]  # x6 = r
    x7 = xk[6]  # x7 = x
    x8 = xk[7]  # x8 = y
    x9 = xk[8]  # x9 = z
    x10 = xk[9]  # x10 = dx/dt
    x11 = xk[10]  # x11 = dy/dt
    x12 = xk[11]  # x12 = dz/dt
    u1[0,k], thetad[0,k], taoq[0,k], phid[0,k], taop[0,k], taor[0,k] = u_last
    
    u1_next = m * g - (dt * (zd - x9) -2.12 * dt * x12 + 0.11 * (m*g-u1[0,k])) / (dt + 0.11)
    
    thetad_next = (opt.kag_1*(xd-x7)-opt.kag_2*x10+opt.kag_3*thetad[0,k]*(-u1_next)/dt)*(dt/(dt+opt.kag_3))/(-u1_next) #kag_1  kag_2  kag_3
    taoq_next = (opt.kt_1*(thetad_next-x2)-opt.kt_2*x5+opt.kt_3*taoq[0,k]/(dt*Iy))*(dt/(dt+opt.kt_3))*Iy             #kt_1  kt_2  kt_3 
    phid_next = (opt.kag_1*(yd-x8)-opt.kag_2*x11+opt.kag_3*phid[0,k]*(u1_next)/dt)*(dt/(dt+opt.kag_3))/(u1_next)       #kag_1  kag_2  kag_3
    taop_next = (opt.kt_1*(phid_next-x1)-opt.kt_2*x4+opt.kt_3*taop[0,k]/(dt*Ix))*(dt/(dt+opt.kt_3))*Ix               #kt_1  kt_2  kt_3 
    
    taor_next = Iz * (opt.kpsi_1 * dt * (psid - x3) - opt.kpsi_2 * dt * x6 + opt.kpsi_3 * taor[0,k]) / (dt + opt.kpsi_3) / Iz  #kpsi_1 kpsi_2  kpsi_3 

    return u1_next, thetad_next, taoq_next, phid_next, taop_next, taor_next 

# 力和力矩到电机控制的转换
def fM2u(f, M):
    mat = np.array([[4.179446268,       4.179446268,        4.179446268,        4.179446268],
                    [-0.6723341164784,  0.6723341164784,    0.6723341164784,    -0.6723341164784],
                    [0.6723341164784,   -0.6723341164784,   0.6723341164784,    -0.6723341164784],
                    [0.055562,          0.055562,           -0.055562,          -0.055562]])
    fM = np.vstack([f, M])
    u = np.dot(np.linalg.inv(mat), fM)
    u1 = u[0, 0]
    u2 = u[1, 0]
    u3 = u[2, 0]
    u4 = u[3, 0]
    return u1, u2, u3, u4

# 欧拉角到旋转矩阵的转换
def angle2R(roll, pitch, yaw):
    sphi = math.sin(roll)
    cphi = math.cos(roll)
    stheta = math.sin(pitch)
    ctheta = math.cos(pitch)
    spsi = math.sin(yaw)
    cpsi = math.cos(yaw)
    R = np.array([[ctheta*cpsi, sphi*stheta*cpsi-cphi*spsi, cphi*stheta*cpsi+sphi*spsi],
                  [ctheta*spsi, sphi*stheta*spsi+cphi*cpsi, cphi*stheta*spsi-sphi*cpsi],
                  [-stheta,     sphi*ctheta,                cphi*ctheta]])
    return R

def get_state(state):
 
    pos_now = np.array([[state.kinematics_estimated.position.x_val],
                        [state.kinematics_estimated.position.y_val],
                        [state.kinematics_estimated.position.z_val]])
    vel_now = np.array([[state.kinematics_estimated.linear_velocity.x_val],
                        [state.kinematics_estimated.linear_velocity.y_val],
                        [state.kinematics_estimated.linear_velocity.z_val]])
    acc_now = np.array([[state.kinematics_estimated.linear_acceleration.x_val],
                        [state.kinematics_estimated.linear_acceleration.y_val],
                        [state.kinematics_estimated.linear_acceleration.z_val]])
    omega_now = np.array([[state.kinematics_estimated.angular_velocity.x_val],
                          [state.kinematics_estimated.angular_velocity.y_val],
                          [state.kinematics_estimated.angular_velocity.z_val]])
    return pos_now, vel_now, acc_now, omega_now

def double_circle_traj():
    p_traj = np.zeros([3, 1600])
    v_traj = np.zeros([3, 1600])
    a_traj = np.zeros([3, 1600])

    for i in range(400):
        theta = math.pi - math.pi / 400 * i
        p_traj[0, i] = 5 * math.cos(theta) + 5
        p_traj[1, i] = 5 * math.sin(theta)
        p_traj[2, i] = -opt.zd
        v_traj[0, i] = 1.9635 * math.cos(theta - math.pi / 2)
        v_traj[1, i] = 1.9635 * math.sin(theta - math.pi / 2)
        v_traj[2, i] = 0
        a_traj[0, i] = -0.7712 * math.cos(theta)
        a_traj[1, i] = -0.7712 * math.sin(theta)
        a_traj[2, i] = 0
    for i in range(400):
        theta = math.pi + math.pi / 400 * i
        p_traj[0, i + 400] = 5 * math.cos(theta) + 15
        p_traj[1, i + 400] = 5 * math.sin(theta)
        p_traj[2, i + 400] = -opt.zd
        v_traj[0, i + 400] = 1.9635 * math.cos(theta + math.pi / 2)
        v_traj[1, i + 400] = 1.9635 * math.sin(theta + math.pi / 2)
        v_traj[2, i + 400] = 0
        a_traj[0, i + 400] = -0.7712 * math.cos(theta)
        a_traj[1, i + 400] = -0.7712 * math.sin(theta)
        a_traj[2, i + 400] = 0
    for i in range(400):
        theta = math.pi / 400 * i
        p_traj[0, i + 800] = 5 * math.cos(theta) + 15
        p_traj[1, i + 800] = 5 * math.sin(theta)
        p_traj[2, i + 800] = -opt.zd
        v_traj[0, i + 800] = 1.9635 * math.cos(theta + math.pi / 2)
        v_traj[1, i + 800] = 1.9635 * math.sin(theta + math.pi / 2)
        v_traj[2, i + 800] = 0
        a_traj[0, i + 800] = -0.7712 * math.cos(theta)
        a_traj[1, i + 800] = -0.7712 * math.sin(theta)
        a_traj[2, i + 800] = 0
    for i in range(400):
        theta = 2 * math.pi - math.pi / 400 * i
        p_traj[0, i + 1200] = 5 * math.cos(theta) + 5
        p_traj[1, i + 1200] = 5 * math.sin(theta)
        p_traj[2, i + 1200] = -opt.zd
        v_traj[0, i + 1200] = 1.9635 * math.cos(theta - math.pi / 2)
        v_traj[1, i + 1200] = 1.9635 * math.sin(theta - math.pi / 2)
        v_traj[2, i + 1200] = 0
        a_traj[0, i + 1200] = -0.7712 * math.cos(theta)
        a_traj[1, i + 1200] = -0.7712 * math.sin(theta)
        a_traj[2, i + 1200] = 0
    return p_traj, v_traj, a_traj

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #任务参数#
    parser.add_argument('--mode', type=str, default='destination')
    parser.add_argument('--xd', type=float, default=2)
    parser.add_argument('--yd', type=float, default=6)
    parser.add_argument('--zd', type=float, default=14)
    parser.add_argument('--psid', type=float, default=0)
    parser.add_argument('--t_max', type=int, default=15)
    parser.add_argument('--dt', type=float, default=0.02)
    #无人机参数#
    parser.add_argument('--g', type=float, default=9.81)
    parser.add_argument('--m', type=float, default=1)
    parser.add_argument('--Ix', type=float, default=8.1 * 1e-3)
    parser.add_argument('--Iy', type=float, default=8.1 * 1e-3)
    parser.add_argument('--Iz', type=float, default=14.2 * 1e-3)
    #控制器参数#  
    #######pd
    parser.add_argument('--ctrl', type=str, default='pd')      
    parser.add_argument('--kp', type=float, default=2)
    parser.add_argument('--kv', type=float, default=2)
    parser.add_argument('--kR', type=float, default=0.4)
    parser.add_argument('--komega', type=float, default=0.08)
    #######超前
    parser.add_argument('--kag_1', type=float, default=1)
    parser.add_argument('--kag_2', type=float, default=2.17)
    parser.add_argument('--kag_3', type=float, default=0.11)
    parser.add_argument('--kt_1', type=float, default=10)
    parser.add_argument('--kt_2', type=float, default=5)
    parser.add_argument('--kt_3', type=float, default=0.11)
    parser.add_argument('--kpsi_1', type=float, default=0.1)
    parser.add_argument('--kpsi_2', type=float, default=0.1)
    parser.add_argument('--kpsi_3', type=float, default=0.02)
    opt=parser.parse_args()
    

    maxtime = opt.t_max
    dt = opt.dt
    t = np.arange(0., maxtime, dt)
    Nk = len(t)
    # 程序开始
    client = airsim.MultirotorClient()
    # 重置位置
    # client.reset()
    time.sleep(2)
    client.enableApiControl(True)
    client.takeoffAsync().join()

    
    if opt.mode == 'destination':
        if opt.ctrl == 'pd':
            p_des = np.array([[opt.xd],[opt.yd],[-opt.zd]])
            v_des = np.array([[0],[0],[0]])
            a_des = np.array([[0],[0],[0]])
            time.sleep(2)
            pos_list = np.zeros((3,400))
            psi_list = np.zeros((1,400))
            
            time.sleep(4)
            #将俯仰角调整至初值为0
            # for i in range(100):
            #     client.moveByRollPitchYawZAsync(0,0,0,-1,0.05).join()
            #     state=client.getMultirotorState()
            #     pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

            for k in range(400):   
                state = client.getMultirotorState()  
                pos_now, vel_now, acc_now, omega_now = get_state(state)
                pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
                R_now = angle2R(roll_now, pitch_now, yaw_now)
                f, M = pd(opt, p_des, v_des, a_des, pos_now, vel_now, R_now, omega_now)
                u1, u2, u3, u4 = fM2u(f, M)
                client.moveByMotorPWMsAsync(u1, u2, u3, u4, 0.05)
                pos_list[:,k]=pos_now.squeeze(1)
                psi_list[:,k]=yaw_now
                
                if k==0:
                    plot_last_pos=[airsim.Vector3r(0, 0, pos_now[2,0])]
                plot_v_start = [airsim.Vector3r(pos_now[0, 0], pos_now[1, 0], pos_now[2, 0])]
                plot_v_end = pos_now + vel_now
                plot_v_end = [airsim.Vector3r(plot_v_end[0, 0], plot_v_end[1, 0], plot_v_end[2, 0])]
                client.simPlotArrows(plot_v_start, plot_v_end, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
                client.simPlotLineList(plot_last_pos + plot_v_start, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)
                plot_last_pos = plot_v_start
                
                time.sleep(0.02) 
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.figure()
            plt.plot(pos_list[0,:],label='x')
            plt.plot(pos_list[1,:],label='y')
            plt.plot(-1*pos_list[2,:],label='z')

            plt.text(250,pos_list[0,-1], "x稳定值{}".format(format(pos_list[0,-1],'.4f')))
            plt.text(250,pos_list[1,-1], "y稳定值{}".format(format(pos_list[1,-1],'.4f')))
            plt.text(250,11.5,             "z稳定值{}".format(format(-1*pos_list[2,-1],'.4f')))
            plt.legend()

            plt.figure()
            plt.plot(psi_list[0,:],label='psi')
            plt.text(250,psi_list[0,-1], "俯仰角稳定值{}".format(format(psi_list[0,-1],'.4f')))

            plt.legend()
            plt.show()   
        elif opt.ctrl == 'lead':           
            thetad = np.zeros([1,Nk])
            phid = np.zeros([1,Nk])
            x = np.zeros([12,Nk])
            u1 = np.zeros([1,Nk])   #拉力uf
            taoq = np.zeros([1,Nk])   #y方向力矩，俯仰角变化，对应u2
            taop= np.zeros([1,Nk])   #x方向力矩，滚转角变化，对应u1
            taor = np.zeros([1,Nk])   #z方向力矩，偏航角变化，对应u3
            pos_list=np.zeros((3,Nk))
            omega_list=np.zeros((3,Nk))
            a_list=np.zeros((3,Nk))
            angle_list=np.zeros((3,Nk)) 
            
            
            #将俯仰角调整至初值为0
            for i in range(100):
                client.moveByRollPitchYawZAsync(0,0,0,2,0.05).join()
                state=client.getMultirotorState()
                pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

            
            for k in range(Nk-1):       
                #读取当前状态
                state = client.getMultirotorState()
                #当前位置
                pos_now = np.array([[state.kinematics_estimated.position.x_val],
                                    [state.kinematics_estimated.position.y_val],
                                    [state.kinematics_estimated.position.z_val]])
                #当前速度
                vel_now = np.array([[state.kinematics_estimated.linear_velocity.x_val],
                                    [state.kinematics_estimated.linear_velocity.y_val],
                                    [state.kinematics_estimated.linear_velocity.z_val]])
                #当前加速度
                acc_now = np.array([[state.kinematics_estimated.linear_acceleration.x_val],
                                    [state.kinematics_estimated.linear_acceleration.y_val],
                                    [state.kinematics_estimated.linear_acceleration.z_val]])
                #当前姿态角速度，注意依次为俯仰，滚转，偏航
                omega_now = np.array([[state.kinematics_estimated.angular_velocity.x_val],
                                    [state.kinematics_estimated.angular_velocity.y_val],
                                    [state.kinematics_estimated.angular_velocity.z_val]])   
                pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation) 
                x[:,k]=[roll_now, pitch_now, yaw_now, omega_now[0,0], omega_now[1,0], omega_now[2,0], \
                        pos_now[0,0], pos_now[1,0], pos_now[2,0], vel_now[0,0], vel_now[1,0], vel_now[2,0]]

                u_last = [u1[0,k], thetad[0,k], taoq[0,k], phid[0,k], taop[0,k], taor[0,k]]
                u1[0,k+1], thetad[0,k+1], taoq[0,k+1], phid[0,k+1], taop[0,k+1], taor[0,k+1] = lead_control(opt, x[:, k], u_last)    
                
                M = np.array([[taop[0,k+1]],[taoq[0,k+1]],[taor[0,k+1]]])
                #力/力矩转换到电机输入
                uc1, uc2, uc3, uc4 = fM2u(u1[0,k+1], M)
                
                angle_list[:,k]=[roll_now, pitch_now, yaw_now]
                pos_list[:,k]=pos_now.squeeze(1)
                omega_list[:,k]=omega_now.squeeze(1)
                a_list[:,k]=acc_now.squeeze(1)
                client.moveByMotorPWMsAsync(uc1, uc2, uc3, uc4, 0.05)
                
                time.sleep(0.02)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.figure()
            plt.plot(pos_list[0,:-1],label='x')
            plt.plot(pos_list[1,:-1],label='y')
            plt.plot(-1*pos_list[2,:-1],label='z')

            plt.text(Nk-150,pos_list[0,Nk-2],    "x稳定值{}".format(format(pos_list[0,-2],'.4f')))
            plt.text(Nk-150,pos_list[1,Nk-2],    "y稳定值{}".format(format(pos_list[1,-2],'.4f')))
            plt.text(Nk-150,-pos_list[2,Nk-2]-2,  "z稳定值{}".format(format(-1*pos_list[2,-2],'.4f')))
            plt.legend()

            plt.figure()
            plt.plot(angle_list[2,:-1],label='psi')
            plt.text(Nk-150,angle_list[2,Nk-2], "俯仰角稳定值{}".format(format(angle_list[2,-2],'.4f')))

            plt.show()
        

    if opt.mode == 'trajectory':
        p_des = np.array([[opt.xd],[opt.yd],[-opt.zd]])
        v_des = np.array([[0],[0],[0]])
        a_des = np.array([[0],[0],[0]])
        time.sleep(2)
        pos_list = np.zeros((3,400))
        psi_list = np.zeros((1,400))
        
        #将俯仰角调整至初值为0
        for i in range(100):
            client.moveByRollPitchYawZAsync(0,0,0,2,0.05).join()
            state=client.getMultirotorState()
            pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
        
        for i in range(400):   
            state = client.getMultirotorState()  
            pos_now, vel_now, acc_now, omega_now = get_state(state)
            print(pos_now)
            pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
            R_now = angle2R(roll_now, pitch_now, yaw_now)
        
            f, M = pd(opt, p_des, v_des, a_des, pos_now, vel_now, R_now, omega_now)
            u1, u2, u3, u4 = fM2u(f, M)
            client.moveByMotorPWMsAsync(u1, u2, u3, u4, 0.05)
            pos_list[:,i]=pos_now.squeeze(1)
            psi_list[:,i]=yaw_now
            time.sleep(0.02) 
               
        time.sleep(2)
        p_traj, v_traj, a_traj=double_circle_traj()
        pos_list = np.zeros((3,1600))
        psi_list = np.zeros((1,1600))
        for i in range(1599):
            plot_v_start = [airsim.Vector3r(p_traj[0, i], p_traj[1, i], p_traj[2, i])]
            plot_v_end = [airsim.Vector3r(p_traj[0, i+1], p_traj[1, i+1], p_traj[2, i+1])]
            client.simPlotLineList(plot_v_start+plot_v_end, color_rgba=[0.0, 1.0, 0.0, 1.0], is_persistent=True)
            
        
        for i in range(1600):   
            state = client.getMultirotorState()  
            pos_now, vel_now, acc_now, omega_now = get_state(state)
            pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
            R_now = angle2R(roll_now, pitch_now, yaw_now)
            if i==0:
                plot_last_pos=[airsim.Vector3r(0, 0, pos_now[2,0])]
            f, M = pd(opt, p_traj[:,i:i+1], v_traj[:,i:i+1], a_traj[:,i:i+1], pos_now, vel_now, R_now, omega_now)
            u1, u2, u3, u4 = fM2u(f, M)
            client.moveByMotorPWMsAsync(u1, u2, u3, u4, 0.05)
            pos_list[:,i]=pos_now.squeeze(1)
            psi_list[:,i]=yaw_now
            plot_v_start = [airsim.Vector3r(pos_now[0, 0], pos_now[1, 0], pos_now[2, 0])]
            plot_v_end = pos_now + vel_now
            plot_v_end = [airsim.Vector3r(plot_v_end[0, 0], plot_v_end[1, 0], plot_v_end[2, 0])]
            client.simPlotArrows(plot_v_start, plot_v_end, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
            client.simPlotLineList(plot_last_pos + plot_v_start, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)
            plot_last_pos = plot_v_start
            time.sleep(0.02)       
    

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure()
        plt.plot(pos_list[0,:], pos_list[1,:], label='实际轨迹')
        plt.plot(p_traj[0,:], p_traj[1,:], label='理想轨迹')
        plt.legend()
        plt.show()


# plt.figure()
# plt.plot(Mx_list[0,:],label='Mx')
# plt.plot(My_list[0,:],label='My')
# plt.plot(Mz_list[0,:],label='Mz')
# plt.legend()
# plt.show()
# waitKey(0)

# plt.figure()
# plt.plot(a_list[0,:],label='ax')
# plt.plot(a_list[1,:],label='ay')
# plt.plot(a_list[2,:],label='az')
# plt.show()
# waitKey(0)

# plt.figure()
# plt.plot(M_list[0,:])
# plt.figure()
# plt.plot(M_list[1,:])
# plt.figure()
# plt.plot(M_list[2,:])
# plt.show()
