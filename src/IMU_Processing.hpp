#ifndef _IMU_PROCESSING_HPP
#define _IMU_PROCESSING_HPP

#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <lidar_imu_init/States.h>
#include <geometry_msgs/Vector3.h>

/// *************Preconfiguration

#define MAX_INI_COUNT (200)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_R_LI_cov(const V3D &R_LI_cov);
  void set_T_LI_cov(const V3D &T_LI_cov);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_mean_acc_norm(const double &mean_acc_norm);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void Process(const MeasureGroup &meas, StatesGroup &state, PointCloudXYZI::Ptr pcl_un_);


//  ros::NodeHandle nh;
  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_R_LI;
  V3D cov_T_LI;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;
  int    lidar_type;
  bool   imu_en;
  bool LI_init_done = false;
  double IMU_mean_acc_norm;

 private:
  void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
  void propagation_and_undist(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_in_out);
  void Forward_propagation_without_imu(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double last_lidar_end_time_;
  double time_last_scan;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true)
{
  imu_en = true;
  init_iter_num = 1;
  cov_acc         = V3D(0.1, 0.1, 0.1);
  cov_gyr         = V3D(0.1, 0.1, 0.1);
  cov_R_LI        = V3D(0.00001, 0.00001, 0.00001);
  cov_T_LI        = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_gyr    = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc    = V3D(0.0001, 0.0001, 0.0001);
  mean_acc        = V3D(0, 0, -1.0);
  mean_gyr        = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  last_imu_.reset(new sensor_msgs::Imu());
  fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}


void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_R_LI_cov(const V3D &R_LI_cov)
{
    cov_R_LI = R_LI_cov;
}

void ImuProcess::set_T_LI_cov(const V3D &T_LI_cov)
{
    cov_T_LI = T_LI_cov;
}

void ImuProcess::set_mean_acc_norm(const double &mean_acc_norm){
    IMU_mean_acc_norm = mean_acc_norm;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
	//进行IMU初始化，根据IMU测量数据来估计重力向量、陀螺仪偏置、加速度计和陀螺仪的协方差，并对加速度计测量进行单位重力归一化
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurements to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100); //初始化进度
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_) //如果是第一帧IMU数据
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z; //平均线速度
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z; //平均角加速度
    first_lidar_time = meas.lidar_beg_time;
  }
  //遍历所有IMU测量数据。获取每个IMU数据的加速度计和陀螺仪测量值。更新加速度计和陀螺仪的均值。更新加速度计和陀螺仪的协方差。
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N ++;
  }

  state_inout.gravity = - mean_acc / mean_acc.norm() * G_m_s2; //计算重力向量
  state_inout.rot_end = Eye3d; //设置旋转矩阵为单位矩阵。
  state_inout.bias_g.setZero(); //将陀螺仪偏置设置为零向量。
  last_imu_ = meas.imu.back();  //更新最后一个IMU数据。
}



void ImuProcess::Forward_propagation_without_imu(const MeasureGroup &meas, StatesGroup &state_inout,
                             PointCloudXYZI &pcl_out) {
    pcl_out = *(meas.lidar);
    /*** sort point clouds by offset time ***/
    const double &pcl_beg_time = meas.lidar_beg_time;
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
    const double &pcl_end_offset_time = pcl_out.points.back().curvature / double(1000);

    MD(DIM_STATE, DIM_STATE) F_x, cov_w;
    double dt = 0.0;

    if (b_first_frame_) {
        dt = 0.1;
        b_first_frame_ = false;
    } else {
        dt = pcl_beg_time - time_last_scan;
        time_last_scan = pcl_beg_time;
    }

    /* covariance propagation */
    F_x.setIdentity();
    cov_w.setZero();
    /** In CV model, bias_g represents angular velocity **/
    /** In CV model，bias_a represents linear acceleration **/
    M3D Exp_f = Exp(state_inout.bias_g, dt);
    F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);
    F_x.block<3, 3>(0, 15) = Eye3d * dt;
    F_x.block<3, 3>(3, 12) = Eye3d * dt;


    cov_w.block<3, 3>(15, 15).diagonal() = cov_gyr_scale * dt * dt;
    cov_w.block<3, 3>(12, 12).diagonal() = cov_acc_scale * dt * dt;

    /** Forward propagation of covariance**/
    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    /** Forward propagation of attitude **/
    state_inout.rot_end = state_inout.rot_end * Exp_f;

    /** Position Propagation **/
    state_inout.pos_end += state_inout.vel_end * dt;

    /**CV model： un-distort pcl using linear interpolation **/
    if(lidar_type != L515){
        auto it_pcl = pcl_out.points.end() - 1;
        double dt_j = 0.0;
        for(; it_pcl != pcl_out.points.begin(); it_pcl --)
        {
            dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
            M3D R_jk(Exp(state_inout.bias_g, - dt_j));
            V3D P_j(it_pcl->x, it_pcl->y, it_pcl->z);
            // Using rotation and translation to un-distort points
            V3D p_jk;
            p_jk = - state_inout.rot_end.transpose() * state_inout.vel_end * dt_j;

            V3D P_compensate =  R_jk * P_j + p_jk;

            /// save Undistorted points and their rotation
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);
        }
    }
}

void ImuProcess::propagation_and_undist(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the current frame-head ***/
  //IMU数据的传播和点云去畸变处理
  pcl_out = *(meas.lidar);
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  double imu_end_time = v_imu.back()->header.stamp.toSec();
  double pcl_beg_time, pcl_end_time;
  //确定激光雷达数据的起始时间和结束时间
  if (lidar_type == L515)
  {
    pcl_beg_time = last_lidar_end_time_;
    pcl_end_time = meas.lidar_beg_time;
  }
  else
  {
    pcl_beg_time = meas.lidar_beg_time;
    /*** sort point clouds by offset time ***/
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
    pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  }


  /*** Initialize IMU pose ***/
  IMUpose.clear();
  //将当前帧的IMU姿态（速度、位置、旋转）作为初始姿态添加到 IMUpose 列表
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));

  /*** forward propagation at each imu point ***/
  /*
  加速度（acc_imu）
  角速度平均值（angvel_avr）
  加速度平均值（acc_avr）
  IMU的速度（vel_imu）
  IMU的位置（pos_imu）
  以及旋转矩阵（R_imu）
  状态转移矩阵（F_x）
  噪声协方差矩阵（cov_w）。
  */
  V3D acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  M3D R_imu(state_inout.rot_end);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  
  double dt = 0;
  //对于每个IMU数据点
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    //1.计算角速度的平均值和加速度的平均值
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

	V3D angvel_now(head->angular_velocity.x, head->angular_velocity.y, head->angular_velocity.z);
	V3D acc_now(head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z);
	fout_imu << setw(10) << head->header.stamp.toSec() << "  " << angvel_now.transpose()<< " " << acc_now.transpose() << endl;
	// 2.校正角速度和加速度，减去偏置值
    angvel_avr -= state_inout.bias_g;
    acc_avr     = acc_avr / IMU_mean_acc_norm * G_m_s2 - state_inout.bias_a;
	// 3.计算时间间隔
    if(head->header.stamp.toSec() < last_lidar_end_time_)
        dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    else
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    
    /* covariance propagation */
    /* 协方差传播 */
    //4.计算相关矩阵（acc_avr_skew）和姿态更新矩阵（Exp_f）。
    M3D acc_avr_skew;
    M3D Exp_f   = Exp(angvel_avr, dt);
    acc_avr_skew<<SKEW_SYM_MATRX(acc_avr);
    // 5. 初始化状态转移矩阵（F_x）和噪声协方差矩阵（cov_w）。
	// 计算相关矩阵和协方差矩阵
    F_x.setIdentity();
    cov_w.setZero();
	// 初始化状态转移矩阵和噪声协方差矩阵
    F_x.block<3,3>(0,0)  = Exp(angvel_avr, -dt);
    F_x.block<3,3>(0,15)  = - Eye3d * dt;
    F_x.block<3,3>(3,12)  = Eye3d * dt;
    F_x.block<3,3>(12,0)  = - R_imu * acc_avr_skew * dt;
    F_x.block<3,3>(12,18) = - R_imu * dt;
    F_x.block<3,3>(12,21) = Eye3d * dt;

    cov_w.block<3,3>(0,0).diagonal()   = cov_gyr * dt * dt;
    cov_w.block<3,3>(6,6).diagonal()   = cov_R_LI * dt * dt;
    cov_w.block<3,3>(9,9).diagonal()   = cov_T_LI * dt * dt;
    cov_w.block<3,3>(12,12)            = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3,3>(15,15).diagonal() = cov_bias_gyr * dt * dt; // bias gyro covariance
    cov_w.block<3,3>(18,18).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance
    // 6. 协方差传播，更新状态的协方差矩阵
    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
	// 7. 更新IMU的加速度、位姿
    /* propagation of IMU attitude (global frame)*/
     // 更新IMU的旋转矩阵（姿态）
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    // 计算全局坐标系IMU的加速度
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propagation of IMU position (global frame)*/
     // 更新全局坐标系IMU的位置
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU (global frame)*/
     // 更新全局坐标系IMU的速度
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements (global frame)*/
    angvel_last = angvel_avr;
    acc_s_last  = acc_imu;
    // 计算时间偏移量
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    //将全局坐标系的姿态信息保存到IMUpose列表中
    IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  //计算出当前帧的速度、位置和姿态的预测值
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  state_inout.vel_end = vel_imu + note * acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;

  //更新最后一个IMU数据和最后一个激光雷达结束时间
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;
  //则执行反向传播对激光雷达数据进行去畸变处理，将激光点从IMU坐标系转换回激光雷达坐标系，并更新激光雷达点云数据。
  if (lidar_type != L515)
  {
    #ifdef DEBUG_PRINT
      cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<endl;
      cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<endl;
    #endif
    /*** un-distort each lidar point (backward propagation) ***/
    auto it_pcl = pcl_out.points.end() - 1; //a single point in k-th frame
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        R_imu << MAT_FROM_ARRAY(head->rot);
        acc_imu << VEC_FROM_ARRAY(head->acc);
        // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        angvel_avr << VEC_FROM_ARRAY(head->gyr);
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = t_j - t_i > 0
            /* Transform to the 'scan-end' IMU frame（I_k frame)*/
            M3D R_i(R_imu * Exp(angvel_avr, dt));
            V3D P_i = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
            V3D p_in(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D P_compensate = state_inout.offset_R_L_I.transpose() * (state_inout.rot_end.transpose() * (R_i * (state_inout.offset_R_L_I * p_in + state_inout.offset_T_L_I) + P_i - state_inout.pos_end) - state_inout.offset_T_L_I);
            /// save Undistorted points
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);
            if (it_pcl == pcl_out.points.begin()) break;
        }
    }
  }
}


void ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  //1. 检查是否启用IMU功能
  if (imu_en)
  {
    if(meas.imu.empty())  return; //确保IMU数据不为空
    ROS_ASSERT(meas.lidar != nullptr);
	 //1. 是否启动IMU初始化
    if (imu_need_init_) 
    {
        if(!LI_init_done){ //如果没有初始化
            /// The very first lidar frame
            IMU_init(meas, stat, init_iter_num); //IMU初始化
            imu_need_init_ = true;
            last_imu_   = meas.imu.back();
            if (init_iter_num > MAX_INI_COUNT)
            {
                cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
                imu_need_init_ = false;

                cov_acc = cov_acc_scale;
                cov_gyr = cov_gyr_scale;

                ROS_INFO("IMU Initialization Done: Gravity: %.4f %.4f %.4f, Acc norm: %.4f", stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm());
                IMU_mean_acc_norm = mean_acc.norm();
            }
        }
        else{
            cout << endl;
            printf(BOLDMAGENTA "[Refinement] Switch to LIO mode, online refinement begins.\n\n" RESET);
            last_imu_   = meas.imu.back();
            imu_need_init_ = false;
            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
        }
        return;
    }
    //向前传播
    propagation_and_undist(meas, stat, *cur_pcl_un_);
  }
  else
  {
     Forward_propagation_without_imu(meas, stat, *cur_pcl_un_);
  }
}
#endif
