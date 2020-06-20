package com.chameleonvision.common.vision.pipe.impl;

import com.chameleonvision.common.logging.LogGroup;
import com.chameleonvision.common.logging.Logger;
import com.chameleonvision.common.vision.pipe.CVPipe;
import com.jogamp.newt.opengl.GLWindow;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.GLBuffers;
import com.jogamp.opengl.util.texture.Texture;
import com.jogamp.opengl.util.texture.TextureData;
import org.opencv.core.*;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class GPUAcceleratedHSVPipe extends CVPipe<Mat, Mat, HSVPipe.HSVParams> {

  private static final String k_vertexShader = String.join("\n",
          "#version 100",
          "",
          "attribute vec4 position;",
          "",
          "void main() {",
          "  gl_Position = position;",
          "}"
  );
  private static final String k_fragmentShader = String.join("\n",
          "#version 100",
          "",
          "precision highp float;",
          "precision highp int;",
          "",
          "uniform vec3 lowerThresh;",
          "uniform vec3 upperThresh;",
          "uniform vec2 resolution;",
          "uniform sampler2D texture0;",
          "",
          "vec3 rgb2hsv(vec3 c) {",
          "  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);",
          "  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));",
          "  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));",
          "",
          "  float d = q.x - min(q.w, q.y);",
          "  float e = 1.0e-10;",
          "  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);",
          "}",
          "",
          "bool inRange(vec3 hsv) {",
          "  bvec3 botBool = greaterThanEqual(hsv, lowerThresh);",
          "  bvec3 topBool = lessThanEqual(hsv, upperThresh);",
          "  return all(botBool) && all(topBool);",
          "}",
          "",
          "void main() {",
          "  vec2 uv = gl_FragCoord.xy/resolution;",
          // Important! We do this .bgr swizzle because the image comes in as BGR but we pretend it's RGB for convenience+speed
          "  vec3 col = texture2D(texture0, uv).bgr;",
          // Only the first value in the vec4 gets used
          "  gl_FragColor = inRange(rgb2hsv(col)) ? vec4(1.0, 0.0, 0.0, 0.0) : vec4(0.0, 0.0, 0.0, 0.0);",
          "}"
  );
  private static final int k_startingWidth = 1280, k_startingHeight = 720;
  private static final float[] k_vertexPositions = {
        // Set up a quad that covers the screen
        -1f, +1f,
        +1f, +1f,
        -1f, -1f,
        +1f, -1f
  };
  private static final int k_positionVertexAttribute = 0; // ID for the vertex shader position variable

  public enum PBOMode {
    NONE,
    SINGLE_BUFFERED,
    DOUBLE_BUFFERED
  }

  private final IntBuffer
          vertexVBOIds = GLBuffers.newDirectIntBuffer(1),
          unpackPBOIds = GLBuffers.newDirectIntBuffer(2),
          packPBOIds = GLBuffers.newDirectIntBuffer(2);

  private final GL2ES2 gl;
  private final GLProfile profile;
  private final PBOMode pboMode;
  private final GLOffscreenAutoDrawable drawable;
  private final Texture texture;
  // The texture uniform holds the image that's being processed
  // The resolution uniform holds the current image resolution
  // The lower and upper uniforms hold the lower and upper HSV limits for thresholding
  private final int textureUniformId, resolutionUniformId, lowerUniformId, upperUniformId;

  private final Logger logger = new Logger(GPUAcceleratedHSVPipe.class, LogGroup.General);

  private byte[] inputBytes, outputBytes;
  private Mat outputMat = new Mat(k_startingHeight, k_startingWidth, CvType.CV_8UC1);
  private int previousWidth = -1, previousHeight = -1;
  private int
          unpackIndex = 0, unpackNextIndex = 0,
          packIndex = 0, packNextIndex = 0;

  public GPUAcceleratedHSVPipe(PBOMode pboMode) {
    this.pboMode = pboMode;

    // Set up GL profile and ask for specific capabilities
    profile = GLProfile.get(pboMode == PBOMode.NONE ? GLProfile.GLES2 : GLProfile.GLES2);
    final var capabilities = new GLCapabilities(profile);
    capabilities.setHardwareAccelerated(true);
    capabilities.setDoubleBuffered(false);
    capabilities.setOnscreen(false);
    capabilities.setRedBits(8);
    capabilities.setBlueBits(8);
    capabilities.setGreenBits(8);
    capabilities.setAlphaBits(0);

    // Set up the offscreen area we're going to draw to
    final var factory = GLDrawableFactory.getFactory(profile);
    drawable = factory.createOffscreenAutoDrawable(factory.getDefaultDevice(), capabilities, new DefaultGLCapabilitiesChooser(), k_startingWidth, k_startingHeight);
    drawable.display();
    drawable.getContext().makeCurrent();

    var sb = new StringBuilder();
    JoglVersion.getDefaultOpenGLInfo(factory.getDefaultDevice(), sb, true);
    System.out.println(sb.toString());

    // Get an OpenGL context; OpenGL ES 2.0 and OpenGL 2.0 are compatible with all the coprocs we care about but not compatible with PBOs. Open GL ES 3.0 and OpenGL 4.0 are compatible with select coprocs *and* PBOs
    gl = pboMode == PBOMode.NONE ? drawable.getGL().getGLES2() : drawable.getGL().getGLES3();
    final int programId = gl.glCreateProgram();

    logger.info("Created an OpenGL context with renderer '" + gl.glGetString(GL.GL_RENDERER) + "', version '" + gl.glGetString(GL.GL_VERSION) + "', and profile '" + profile.toString() + "'");

    var fmt = GLBuffers.newDirectIntBuffer(1);
    gl.glGetIntegerv(GL4ES3.GL_IMPLEMENTATION_COLOR_READ_FORMAT, fmt);
    var type = GLBuffers.newDirectIntBuffer(1);
    gl.glGetIntegerv(GL4ES3.GL_IMPLEMENTATION_COLOR_READ_TYPE, type);

    logger.info("GL_IMPLEMENTATION_COLOR_READ_FORMAT: " + fmt.get(0) + ", GL_IMPLEMENTATION_COLOR_READ_TYPE: " + type.get(0));

    // Tell OpenGL that the attribute in the vertex shader named position is bound to index 0 (the index for the generic position input)
    gl.glBindAttribLocation(programId, 0, "position");

    // Compile and setup our two shaders with our program
    final int vertexId = createShader(gl, programId, k_vertexShader, GL2ES2.GL_VERTEX_SHADER);
    final int fragmentId = createShader(gl, programId, k_fragmentShader, GL2ES2.GL_FRAGMENT_SHADER);

    // Link our program together and check for errors
    gl.glLinkProgram(programId);
    IntBuffer status = GLBuffers.newDirectIntBuffer(1);
    gl.glGetProgramiv(programId, GL2ES2.GL_LINK_STATUS, status);
    if (status.get(0) == GL2ES2.GL_FALSE) {

      IntBuffer infoLogLength = GLBuffers.newDirectIntBuffer(1);
      gl.glGetProgramiv(programId, GL2ES2.GL_INFO_LOG_LENGTH, infoLogLength);

      ByteBuffer bufferInfoLog = GLBuffers.newDirectByteBuffer(infoLogLength.get(0));
      gl.glGetProgramInfoLog(programId, infoLogLength.get(0), null, bufferInfoLog);
      byte[] bytes = new byte[infoLogLength.get(0)];
      bufferInfoLog.get(bytes);
      String strInfoLog = new String(bytes);

      throw new RuntimeException("Linker failure: " + strInfoLog);
    }
    gl.glValidateProgram(programId);

    // Cleanup shaders that are now compiled in
    gl.glDetachShader(programId, vertexId);
    gl.glDetachShader(programId, fragmentId);
    gl.glDeleteShader(vertexId);
    gl.glDeleteShader(fragmentId);

    // Tell OpenGL to use our program
    gl.glUseProgram(programId);

    // Set up our texture
    textureUniformId = gl.glGetUniformLocation(programId, "texture0");
    texture = new Texture(GL2ES2.GL_TEXTURE_2D);
    texture.setTexParameteri(gl, GL2ES2.GL_TEXTURE_MIN_FILTER, GL2ES2.GL_LINEAR);
    texture.setTexParameteri(gl, GL2ES2.GL_TEXTURE_MAG_FILTER, GL2ES2.GL_LINEAR);
    texture.setTexParameteri(gl, GL2ES2.GL_TEXTURE_WRAP_S, GL2ES2.GL_CLAMP_TO_EDGE);
    texture.setTexParameteri(gl, GL2ES2.GL_TEXTURE_WRAP_T, GL2ES2.GL_CLAMP_TO_EDGE);

    // Set up a uniform to hold image resolution
    resolutionUniformId = gl.glGetUniformLocation(programId, "resolution");

    // Set up uniforms for the HSV thresholds
    lowerUniformId = gl.glGetUniformLocation(programId, "lowerThresh");
    upperUniformId = gl.glGetUniformLocation(programId, "upperThresh");

    // Set up a quad that covers the entire screen so that our fragment shader draws onto the entire screen
    gl.glGenBuffers(1, vertexVBOIds);

    FloatBuffer vertexBuffer = GLBuffers.newDirectFloatBuffer(k_vertexPositions);
    gl.glBindBuffer(GL2ES2.GL_ARRAY_BUFFER, vertexVBOIds.get(0));
    gl.glBufferData(GL2ES2.GL_ARRAY_BUFFER, vertexBuffer.capacity() * Float.BYTES, vertexBuffer, GL2ES2.GL_STATIC_DRAW);

    // Set up pixel unpack buffer (a PBO to transfer image data to the GPU)
    if (pboMode != PBOMode.NONE) {
      gl.glGenBuffers(2, unpackPBOIds);

      gl.glBindBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER, unpackPBOIds.get(0));
      gl.glBufferData(GL4ES3.GL_PIXEL_UNPACK_BUFFER, k_startingHeight * k_startingWidth * 3, null, GL4ES3.GL_STREAM_DRAW);
      if (pboMode == PBOMode.DOUBLE_BUFFERED) {
        gl.glBindBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER, unpackPBOIds.get(1));
        gl.glBufferData(GL4ES3.GL_PIXEL_UNPACK_BUFFER, k_startingHeight * k_startingWidth * 3, null, GL4ES3.GL_STREAM_DRAW);
      }
      gl.glBindBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER, 0);
    }

    // Set up pixel pack buffer (a PBO to transfer the processed image back from the GPU)
    if (pboMode != PBOMode.NONE) {
      gl.glGenBuffers(2, packPBOIds);

      gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, packPBOIds.get(0));
      gl.glBufferData(GL4ES3.GL_PIXEL_PACK_BUFFER, k_startingHeight * k_startingWidth, null, GL4ES3.GL_STREAM_READ);
      if (pboMode == PBOMode.DOUBLE_BUFFERED) {
        gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, packPBOIds.get(1));
        gl.glBufferData(GL4ES3.GL_PIXEL_PACK_BUFFER, k_startingHeight * k_startingWidth, null, GL4ES3.GL_STREAM_READ);
      }
      gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, 0);
    }

  }

  private static int createShader(GL2ES2 gl, int programId, String glslCode, int shaderType) {
    int shaderId = gl.glCreateShader(shaderType);
    if (shaderId == 0)
      throw new RuntimeException("Shader ID is zero");

    IntBuffer length = GLBuffers.newDirectIntBuffer(new int[]{glslCode.length()});
    gl.glShaderSource(shaderId, 1, new String[] {glslCode}, length);
    gl.glCompileShader(shaderId);

    IntBuffer intBuffer = IntBuffer.allocate(1);
    gl.glGetShaderiv(shaderId, GL2ES2.GL_COMPILE_STATUS, intBuffer);

    if (intBuffer.get(0) != 1) {
      gl.glGetShaderiv(shaderId, GL2ES2.GL_INFO_LOG_LENGTH, intBuffer);
      int size = intBuffer.get(0);
      if (size > 0) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(size);
        gl.glGetShaderInfoLog(shaderId, size, intBuffer, byteBuffer);
        System.err.println(new String(byteBuffer.array()));
      }
      throw new RuntimeException("Couldn't compile shader");
    }

    gl.glAttachShader(programId, shaderId);

    return shaderId;
  }

  @Override
  protected Mat process(Mat in) {
    if (in.width() != previousWidth && in.height() != previousHeight) {
      logger.debug("Resizing OpenGL viewport, byte buffers, and PBOs");

      drawable.setSurfaceSize(in.width(), in.height());
      gl.glViewport(0, 0, in.width(), in.height());

      previousWidth = in.width();
      previousHeight = in.height();

      inputBytes = new byte[in.width() * in.height() * 3];

      outputBytes = new byte[in.width() * in.height()];
      outputMat = new Mat(k_startingHeight, k_startingWidth, CvType.CV_8UC1);

      if (pboMode != PBOMode.NONE) {
        gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, packPBOIds.get(0));
        gl.glBufferData(GL4ES3.GL_PIXEL_PACK_BUFFER, in.width() * in.height(), null, GL4ES3.GL_STREAM_READ);

        if (pboMode == PBOMode.DOUBLE_BUFFERED) {
          gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, packPBOIds.get(1));
          gl.glBufferData(GL4ES3.GL_PIXEL_PACK_BUFFER, in.width() * in.height(), null, GL4ES3.GL_STREAM_READ);
        }
      }
    }

    if (pboMode == PBOMode.DOUBLE_BUFFERED) {
      unpackIndex = (unpackIndex + 1) % 2;
      unpackNextIndex = (unpackIndex + 1) % 2;
    }

    // Reset the fullscreen quad
    gl.glBindBuffer(GL2ES2.GL_ARRAY_BUFFER, vertexVBOIds.get(0));
    gl.glEnableVertexAttribArray(k_positionVertexAttribute);
    gl.glVertexAttribPointer(0, 2, GL2ES2.GL_FLOAT, false, 0, 0);

    // Load and bind our image as a 2D texture
    gl.glActiveTexture(GL2ES2.GL_TEXTURE0);
    texture.enable(gl);
    texture.bind(gl);

    // Load our image into the texture
    in.get(0, 0, inputBytes);
    if (pboMode == PBOMode.NONE || true) {
      ByteBuffer buf = ByteBuffer.wrap(inputBytes);
      // (We're actually taking in BGR even though this says RGB; it's much easier and faster to switch it around in the fragment shader)
      texture.updateImage(gl, new TextureData(profile, GL2ES2.GL_RGB8, in.width(), in.height(), 0, GL2ES2.GL_RGB, GL2ES2.GL_UNSIGNED_BYTE, false, false, false, buf, null));
    } else {
      // Bind the PBO to the texture
      gl.glBindBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER, unpackPBOIds.get(unpackIndex));

      // Copy pixels from the PBO to the texture object
      gl.glTexSubImage2D(GL4ES3.GL_TEXTURE_2D, 0, 0, 0, in.width(), in.height(), GL4ES3.GL_RGB8, GL4ES3.GL_UNSIGNED_BYTE, 0);

      // Bind (potentially) another PBO to update the texture source
      gl.glBindBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER, unpackPBOIds.get(unpackNextIndex));

      // This call with a nullptr for the data arg tells OpenGL *not* to wait to be in sync with the GPU
      // This causes the previous data in the PBO to be discarded
      gl.glBufferData(GL4ES3.GL_PIXEL_UNPACK_BUFFER, in.width() * in.height() * 3, null, GL4ES3.GL_STREAM_DRAW);

      // Map the a buffer of GPU memory into a place that's accessible by us
      var buf = gl.glMapBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER, GL4ES3.GL_WRITE_ONLY);
      buf.put(inputBytes);

      gl.glUnmapBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER);
      gl.glBindBuffer(GL4ES3.GL_PIXEL_UNPACK_BUFFER, 0);
    }

    // Set up a uniform holding our image as a texture
    gl.glUniform1i(textureUniformId, 0);

    // Set up a uniform holding the image resolution
    gl.glUniform2f(resolutionUniformId, in.width(), in.height());

    // Set up threshold uniforms
    var lowr = params.getHsvLower().val;
    var upr = params.getHsvUpper().val;
    gl.glUniform3f(lowerUniformId, (float) lowr[0], (float) lowr[1], (float) lowr[2]);
    gl.glUniform3f(upperUniformId, (float) upr[0], (float) upr[1], (float) upr[2]);

    // Draw the fullscreen quad
    gl.glDrawArrays(GL2ES2.GL_TRIANGLE_STRIP, 0, k_vertexPositions.length);

    // Cleanup
    texture.disable(gl);
    gl.glDisableVertexAttribArray(k_positionVertexAttribute);
    gl.glUseProgram(0);

    if (pboMode == PBOMode.NONE) {
      return saveMatNoPBO(gl, in.width(), in.height());
    } else {
      return saveMatPBO(new DebugGLES3((GLES3) gl), in.width(), in.height(), pboMode == PBOMode.DOUBLE_BUFFERED);
    }
  }

  private static Mat saveMatNoPBO(GL2ES2 gl, int width, int height) {
    ByteBuffer buffer = GLBuffers.newDirectByteBuffer(width * height);
    // We use GL_RED to get things in a single-channel format
    // Note that which pixel format you use is *very* important to performance
    // E.g. GL_LUMINANCE is super slow in this case
    gl.glReadPixels(0, 0, width, height, GL2ES2.GL_RED, GL2ES2.GL_UNSIGNED_BYTE, buffer);

    return new Mat(height, width, CvType.CV_8UC1, buffer);
  }

  private Mat saveMatPBO(GL4ES3 gl, int width, int height, boolean doubleBuffered) {
    if (doubleBuffered) {
      packIndex = (packIndex + 1) % 2;
      packNextIndex = (packIndex + 1) % 2;
    }

    // Set the target framebuffer to read
//    gl.glReadBuffer(GL4ES3.GL_FRONT);

    // Read pixels from the framebuffer to the PBO
    gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, packPBOIds.get(packIndex));
    // We use GL_RED to get things in a single-channel format
    // Note that which pixel format you use is *very* important to performance
    // E.g. GL_LUMINANCE is super slow in this case
    // TODO: GL_RED + GL_UNSIGNED_BYTE may not be supported on most OpenGL ES 3.0 implementations
    gl.glReadPixels(0, 0, width, height, GL4ES3.GL_RED, GL4ES3.GL_UNSIGNED_BYTE, 0);

    // Map the PBO into the CPU's memory
    gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, packPBOIds.get(packNextIndex));
    var buf = gl.glMapBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, GL4ES3.GL_READ_ONLY);
    buf.get(outputBytes);
    outputMat.put(0, 0, outputBytes);
    gl.glUnmapBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER);
    gl.glBindBuffer(GL4ES3.GL_PIXEL_PACK_BUFFER, 0);

    return outputMat;
  }
}
