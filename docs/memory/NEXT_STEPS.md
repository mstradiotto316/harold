# Harold Next Steps

1. Run a real world hardware test with the robot off of the test stand under its own weight to verify all the changes landed correctly and did not result in regressions to the walking pattern. (Ensure the IMU is recording data)
2. Take the logs from the real world hardware test and copy them from the robot's raspberry pi to the Desktop computer
3. Run the simulated walking and compare the commands from the real robot to the simulated robot. If they do not match, update the simulated robot settings (stiffness and dampening) until the sim matches the real hardware exactly.
4. Run the simulated walking and compare the actual joint positions from the real robot to the simulated robot. If they do not match, update the simulated robot settings (stiffness and dampening) until the sim matches the real hardware within a realistic margin of error.
5. Run the simulated walking and compare the actual IMU data from the real robot's IMU data. If they do not match, attempt to add noise to simulated IMU data until it resonably matches what we see in the real world. If the match is impossible, consider ways we can clean or get better data from the real robot. Work with me to plan out a strategy.
