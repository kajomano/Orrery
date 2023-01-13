# Accelerated Raytracing in Python with Pytorch

The original idea was to accelerate raytracing using vectorized coherent-ray calculations with pytorch, but it is clear now that the GPU can be much better utilized if a separate computational node handles every single ray.

I continued a Vulkan-based raytracer in the Kea repository (the Kea name was chosen because it is a volcano as well as a famous observatory :)).

![](orrery_rt_image.png)
