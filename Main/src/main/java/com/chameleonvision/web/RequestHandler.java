package com.chameleonvision.web;

import com.chameleonvision.config.ConfigManager;
import com.chameleonvision.network.NetworkIPMode;
import com.chameleonvision.vision.VisionManager;
import com.chameleonvision.vision.VisionProcess;
import com.chameleonvision.vision.camera.CameraCapture;
import com.chameleonvision.vision.enums.StreamDivisor;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.javalin.http.Context;

import java.util.Map;

public class RequestHandler {

    public static void onGeneralSettings(Context ctx) {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            Map map = objectMapper.readValue(ctx.body(), Map.class);

            // TODO: change to function, to restart NetworkTables
            ConfigManager.settings.teamNumber = (int) map.get("teamNumber");

            ConfigManager.settings.connectionType = NetworkIPMode.values()[(int) map.get("connectionType")];
            ConfigManager.settings.ip = (String) map.get("ip");
            ConfigManager.settings.netmask = (String) map.get("netmask");
            ConfigManager.settings.gateway = (String) map.get("gateway");
            ConfigManager.settings.hostname = (String) map.get("hostname");
            ConfigManager.saveGeneralSettings();
            SocketHandler.sendFullSettings();
            ctx.status(200);
        } catch (JsonProcessingException e) {
            ctx.status(500);
        }
    }

    public static void onCameraSettings(Context ctx) {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            Map camSettings = objectMapper.readValue(ctx.body(), Map.class);

            VisionProcess currentVisionProcess = VisionManager.getCurrentUIVisionProcess();
            CameraCapture currentCamera = currentVisionProcess.getCamera();

            Double newFOV = (Double) camSettings.get("fov");
            Integer newStreamDivisor = (Integer) camSettings.get("streamDivisor");
            Integer newResolution = (Integer) camSettings.get("resolution");

            currentCamera.getProperties().setFOV(newFOV);

            currentVisionProcess.cameraStreamer.setDivisor(StreamDivisor.values()[newStreamDivisor], true);

            currentCamera.setVideoMode(newResolution);

            VisionManager.saveCurrentCameraSettings();
            SocketHandler.sendFullSettings();
            ctx.status(200);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            ctx.status(500);
        }
    }
}