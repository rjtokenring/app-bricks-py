# Video Object Tracking Brick

This Brick provides a Python interface for **detecting objects in real time from a USB camera video stream**.  
It connects to a model runner over WebSocket, continuously analyzes incoming frames, and produces detection events with predicted labels, bounding boxes, and confidence scores.  

Beyond visualization, it allows you to **register callbacks** that react to detections, either for specific objects or for all detections, enabling event-driven logic in your applications.  
It supports both **pre-trained models** provided by the framework and **custom models** trained with Edge Impulse.

## Overview

TODO