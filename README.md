# 🤖 Basics of Mobile Robotics Project - The Thytanic Navigation 

## Project Description

This project involves guiding the **Thytanic** from its starting position to its destination in an efficient manner using `Global Navigation`. Along the way, the Thytanic must avoid appearing obstacles by employing `Local Navigation`. Additionally, the position of the Thytanic is estimated using a `Kalman Filter`. Finally, a stationary camera is positioned above the map to provide `Vision` information for the system’s navigation modules.

Imagine the following scenario: the Thytanic is navigating in the vast ocean (**blue background**) and is seeking to reach land (**white area**) safely while avoiding dangerous icebergs (**local obstacles**). At the same time, it must avoid colliding with **black zones** representing dangerous land masses. Its goal is to navigate the shortest, most energy-efficient route to safety.

![Thytanic](img/thytanic.png)
