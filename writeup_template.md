## Project 2: Kinematics Pick & Place (KUKA KR210)
### Author: Anand Kannan
---


**Steps taken to complete the project:**


1. Set up ROS Workspace.
2. Cloned the [project repository](https://github.com/udacity/RoboND-Kinematics-Project) into the ***src*** directory of  ROS Workspace.
3. Experimented with the forward_kinematics environment and familiarized myself with the robot.
4. Launch in [demo mode](https://classroom.udacity.com/nanodegrees/nd209/parts/7b2fd2d7-e181-401e-977a-6158c77bf816/modules/8855de3f-2897-46c3-a805-628b5ecf045b/lessons/91d017b1-4493-4522-ad52-04a74a01094c/concepts/ae64bb91-e8c4-44c9-adbe-798e8f688193).
5. Perform Kinematic Analysis for the robot following the [project rubric](https://review.udacity.com/#!/rubrics/972/view).
6. Fill in the `IK_server.py` with your Inverse Kinematics code. 


[//]: # (Image References)

[image1]: ./misc_images/misc1.png
[image2]: ./misc_images/misc3.png
[image3]: ./misc_images/misc2.png
[c1]: ./misc_images/comp1.png
[c2]: ./misc_images/comp2.png
[c3]: ./misc_images/comp3.png
[c4]: ./misc_images/comp4.png
[dhpic]: ./misc_images/pic1.jpg
[angpic]: ./misc_images/pic2.jpg
[angpic2]: ./misc_images/pic3.jpg
[eq1]: ./misc_images/1.png
[eq2]: ./misc_images/2.png
[eq3]: ./misc_images/3.png
[eq4]: ./misc_images/4.png


---
### Writeup
In order to successfully complete this project, I used the following environment:
- Ubuntu 16.04
- ROS Kinetic
- Gazebo 7
- RVIZ 1.12.16


### Kinematic Analysis
#### 1. Running Forward Kinematics Demo
I ran the forward kinematics demo as shown below to familarize myself with the RVIZ environment, and to see how each joint reacts when the joint space parameters were adjusted and the direction and orientation of each joint in the cartesian space. To run the demo, I ran the following shell script: 

```sh
$ roslaunch kuka_arm forward_kinematics.launch
```

![alt text][image1]

#### 2. Obtaining DH Parameter Table

We have to first extract the positions and orientation for each joint using the URDF file provided, `kr210.urdf.xacro`. Shown below is a sketch of the KUKA arms joints and links, and relevant parameters for the Modified DH parameter table where:

**a(i-1)**: Link Length from Z(i-1) to Z(i) along to X(i-1) axis

**d(i)**: Link Offset from X(i-1) to X(i) along Z(i) axis

**alpha(i-1)**: Twist Angle from Z(i-1) to Z(i) measured about X(i-1) axis

**theta(i)**: Joint Angle from X(i-1) to X(i) measured about Z(i) axis


![alt text][dhpic]


**Modified DH Parameter Table**

Links | alpha(i-1) | a(i-1) | d(i) | theta(i)
:---: | :---:     | :---:    | :---:  | :---:
0-1 | 0       | 0      | 0.75 | qi
1-2 | - pi/2  | 0.35   | 0    | - pi/2 + q2
2-3 | 0       | 1.25   | 0    | q3
3-4 |  - pi/2 | -0.054 | 1.50 | q4
4-5 | pi/2    | 0      | 0    | q5
5-6 | - pi/2  | 0      | 0    | q6
6-EE| 0       | 0      | 0.303| 0

To get transformation matrices between each frame, I created the function below which takes input of `alpha`, `a`, `d`, and `q`:

```sh
def TF_Matrix(alpha,a,d,q):
	    TF = Matrix([[	cos(q),		 -sin(q),	   0,            a],
		    [sin(q)*cos(alpha),cos(q)*cos(alpha),-sin(alpha),-sin(alpha)*d],
		    [sin(q)*sin(alpha),cos(q)*sin(alpha), cos(alpha), cos(alpha)*d],
		    [		     0,		       0,	   0,		1]])
	    return TF
```

Then, to derive the tranformation matrix between each link, run the above function with the appropriate inputs:

```sh
T0_1 = TF_Matrix(alpha0, a0, d1, q1).subs(DH_Table)
T1_2 = TF_Matrix(alpha1, a1, d2, q2).subs(DH_Table)
T2_3 = TF_Matrix(alpha2, a2, d3, q3).subs(DH_Table)
T3_4 = TF_Matrix(alpha3, a3, d4, q4).subs(DH_Table)
T4_5 = TF_Matrix(alpha4, a4, d5, q5).subs(DH_Table)
T5_6 = TF_Matrix(alpha5, a5, d6, q6).subs(DH_Table)
T6_EE = TF_Matrix(alpha6, a6, d7, q7).subs(DH_Table)
```
The transformation Matrix for each joint is given below:

`T0_1`:
```sh
[cos(q1), -sin(q1), 0, 0],
[sin(q1),  cos(q1), 0, 0],
[      0,        0, 1, 0.75],
[      0,        0, 0, 1]
```

`T1_2`:
```sh
[ cos(q2 - 0.5*pi), -sin(q2 - 0.5*pi), 0, 0.35],
[                0,                 0, 1, 0],
[-sin(q2 - 0.5*pi), -cos(q2 - 0.5*pi), 0, 0],
[                0,                 0, 0, 1]
```

`T2_3`:
```sh
[cos(q3), -sin(q3), 0, 1.25],
[sin(q3),  cos(q3), 0, 0],
[      0,        0, 1, 0],
[      0,        0, 0, 1]
```

`T3_4`:
```sh
[[cos(q4), -sin(q4), 0, -0.054],
[       0,        0, 1, 1.5],
[-sin(q4), -cos(q4), 0, 0],
[       0,        0, 0, 1]]
```

`T4_5`:
```sh
[[cos(q5), -sin(q5), 0, 0],
[       0,        0,-1, 0],
[ sin(q5),  cos(q5), 0, 0],
[       0,        0, 0, 1]]
```

`T5_6`:
```sh
[ cos(q6), -sin(q6), 0, 0],
[       0,        0, 1, 0],
[-sin(q6), -cos(q6), 0, 0],
[       0,        0, 0, 1]
```

`T6_EE`:
```sh
[1, 0, 0, a6],
[0, 1, 0, 0],
[0, 0, 1, 0.303],
[0, 0, 0, 1]
```

Finally, to get the generalized homogenous transform between base_link and the gripper_link, simply multiply all the transforms like such:

```sh
T0_EE = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_EE
```
which results to the matrix below when all the q values are equal to 0:

```sh
[0, 0, 1, 2.153],
[0,-1, 0, 0],
[1, 0, 0, 1.946],
[0, 0, 0, 1]
```

To correct for the orientation difference of the gripper link  between URDF and DH convention, we need a matrix that rotates first about the y axis then the z axis. I implemented the transform below to do so:

```sh
    Rerror = ROT_z.subs(y,radians(180))*ROT_y.subs(p,radians(-90))
```


#### 3. Calculating Joint Angles

The final three joints in the KUKA KR210 robot that we use are all revolute and since thier joint axes intersect at a single point, we can identify this set of joints as a spherical wrist with the wrist center (WC) at Joint 5. We can now separate this Inverse Kinematics problem into Inverse Position and Inverse Orientation problems.

##### Inverse Position

In order to get the WC position, we need the end effector (EE) position and orientation, which we can obtain through with the code shown below:

```sh
    px = req.poses[x].position.x
    py = req.poses[x].position.y
    pz = req.poses[x].position.z
   
    EE = Matrix([[px],
                 [py],
                 [pz]])
   
    (roll,pitch,yaw) = tf.transformations.euler_from_quaternion(
        [req.poses[x].orientation.x,
         req.poses[x].orientation.y,
         req.poses[x].orientation.z,
         req.poses[x].orientation.w])
``` 
Then, we can build a rotation matrix for the EE and apply the correction matrix for the difference between URDF and DH conventions:


```sh
	ROT_x = Matrix([[1,     0,      0],
	    	        [0,cos(r),-sin(r)],
		        [0,sin(r), cos(r)]]) #ROLL

	ROT_y = Matrix([[cos(p) ,	0, sin(p)],
		        [0      ,	1,      0],
		        [-sin(p), 	0,cos(p)]]) # PITCH

	ROT_z = Matrix([[cos(y),-sin(y),	0],
    		        [sin(y), cos(y),	0],
		        [0     ,      0,    1]]) # YAW
 
	ROT_EE = ROT_z*ROT_y*ROT_x

	Rerror = ROT_z.subs(y,radians(180))*ROT_y.subs(p,radians(-90))

	ROT_EE = ROT_EE*Rerror

   	ROT_EE = ROT_EE.subs({'r':roll,'p':pitch,'y':yaw})
```
The x, y, and z coordinates for the WC can now be calculated:

```sh
        WC = EE - (0.303)*ROT_EE[:,2]
```


Now, we can easily calculate `theta1`, the angle of the Joint 1 by simply doing `arctan` of the x and y coordinates of the WC:

```sh
	theta1 = atan2(WC[1],WC[0])
```

In order to calculate `theta2` and `theta3`, refer to the diagram below:

![alt text][angpic2]

Then, we can use the law of cosines to calculate angles `a`, `b`, and `c`:

```sh
    angle_a = acos((side_b*side_b+side_c*side_c-side_a*side_a)/(2*side_b*side_c))
    angle_b = acos((side_a*side_a+side_c*side_c-side_b*side_b)/(2*side_a*side_c))
    angle_c = acos((side_b*side_b+side_a*side_a-side_c*side_c)/(2*side_b*side_a))
```

Now that we have obtained values for angles `a`, `b`, and `c`, we can calculate for `theta2` and `theta3`:

```sh
    theta2 = pi/2 - angle_a -atan2(WC[2]-0.75, sqrt(WC[0]*WC[0]+WC[1]*WC[1])-.35)
    theta3 = pi/2 - (angle_b+.036)
```

##### Inverse Orientation
Now that we have values for `theta1`, `theta2`, and `theta3`, we need to find the remaining three angles. The product of the rotations about each joint must be equal to the roll, pitch, and yaw between the `base_link` and `gripper_link`. Therefore we can say that:

```sh
    R0_6 = ROT_EE
```

We can subsitute our values for the first three angles and multiply each side by `inv(R0_3)` and we can determine the specific rotation matrix from Link 3 to 6:

```sh
	    R0_3 = T0_1[0:3,0:3]*T1_2[0:3,0:3]*T2_3[0:3,0:3]
	    R0_3 = R0_3.evalf(subs={q1:theta1, q2:theta2, q3:theta3})
		
	    R3_6 = R0_3.inv("LU")*ROT_EE
```

And then we can finally solve for `theta4`, `theta5`, and `theta6`. These three angles are the Euler Angles of the rotation matrix `R3_6` shown above. This matrix is determined by breaking down the overall rotation of the end effector as such: `R0_3 * R3_6 = ROT_EE`
Refering to the images below, `theta4`, `theta5`, and `theta6` corresponds to gamma, beta, and alpha, respectively.

![alt text][eq1]
![alt text][eq2]
![alt text][eq3]
![alt text][eq4]


```sh
    r11 = R3_6[0,0]
    r21 = R3_6[1,0]
    r31 = R3_6[2,0]
    r32 = R3_6[2,1]
    r33 = R3_6[2,2]


    theta6 = atan2( r21, r11).evalf()
    theta5 = atan2(-r31, sqrt(r11**2 + r21**2)).evalf()
    theta4 = atan2( r32, r33).evalf()
```

### Project Implementation

#### 1. Code Description And Results 

The code is essentially a tool to help calculate the transformation matrices, perform inverse orientation and position problems, and to visualize the movement of the robot in the simulation. My code starts with importing the important modules, followed by initiallizing the symbols required to perform the kinematic analysis of this project. The code runs a service called `kuka_arm.srv` and uses `trajectory_msgs.msg` and `geometry_msgs.msg` to communicate with the robot. The code, `IK_server.py` can be found in the `scripts` folder of the `kuka_arm` directory. 

Using the roll, pitch, yaw, and positions of the end effector, we can determine the WC position. Then, we can solve for `theta_2` and `theta_3` as an inverse position problem as described above.

Using the rotation matrix only between Links 0 and 3 and the complete rotational matrix between the base_link and the end_effector, we can use the `inv` function to find the rotation matrix between Links 3 and 6. Finally, we can calculate `theta_4`, `theta_5`, and `theta_6`.

For the results, the arm was able to place exactly 8/10 of the targets in the cylinder. I noticed that when the targets were on the bottom shelf, the gripper would knock it down before it could grasp it and therefore was unable to successfully complete the process. It consistently picks up the targets easily from the other shelves. The speed of the arm is very slow. As you can see from the screenshot below, it took me over an hour (real time) for 10 tests. This most likely had to do with running it on my slow laptop and on the VM. It took about 30 minutes of simulation time to perform these tests.

All these joint angles are stored in the `joint_state_publisher` node and is sent to RVIZ to move the arm. Depending on the path of the arm, it will have several joint states for which the `IK_server` returns the joint angles for each one. The lower the amount fo joint states, the faster the arm will be.

To further improve the implementation of the `IK_server` code, we can try to optimize the planning code to include only key pointand to minimize any extra movement from the arm.

And here are some images of the completed pick and place process:
![alt text][c4]

