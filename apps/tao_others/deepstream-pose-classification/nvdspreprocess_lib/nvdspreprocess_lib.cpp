/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file  nvdspreprocess_lib.cpp
 * @brief Custom Library Implementation
 */

// Header includes for standard and system libraries
#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>
#include <sys/time.h>
#include <vector>

// Header includes for NVIDIA libraries
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

// Header includes for NVIDIA DeepStream libraries
#include "nvbufsurface.h"
#include "nvdspreprocess_lib.h"
#include "nvdsmeta_schema.h"

// Namespace declarations
using std::mutex;
using std::unique_lock;
using std::vector;
using ObjectIterator = vector<SObjectContex *>::iterator;

// Macro Definitions for configurable parameters.
#define _MIN_FRAME_ 3
#define _MAX_FRAME_ 300
#define _MAX_OBJECT_NUM_ 20
#define _TIME_OUT_ 2

// Macro for freeing memory.
#define FREE(p) (free(p), p = NULL)

/** @struct SObjectContex
 *  @brief Structure to represent object context holding keypoints information.
 */
struct SObjectContex
{
  uint64_t object_id; // Unique identifier for the object
  float *x, *y, *z;   // Pointers to arrays for x, y, and z coordinates of keypoints
  int frameIndex;     // Current frame index
  bool firstUse;      // Flag indicating whether the object is being used for the first time
  long tv_sec;        // Seconds part of the timestamp

  // Constructor: Initializes the object with default values.
  SObjectContex()
  {
    object_id = UNTRACKED_OBJECT_ID;
    x = y = z = NULL;
    frameIndex = 0;
    firstUse = true;
    tv_sec = 0;
  };

  // Destructor: Frees the allocated memory for x, y, and z arrays.
  ~SObjectContex()
  {
    FREE(x);
    FREE(y);
    FREE(z);
  };
};

/** @struct CustomCtx
 *  @brief Structure to hold custom context data.
 */
struct CustomCtx
{
  std::mutex mtx;                             // Mutex for synchronization
  std::vector<SObjectContex *> multi_objects; // Vector to store SObjectContex objects
  int one_channel_element_num;                // Number of elements in one channel
  int two_channel_element_num;                // Number of elements in two channels
  int move_element_num;                       // Number of elements for moving keypoints

  // Destructor: Frees the allocated memory for SObjectContex objects in multi_objects vector.
  ~CustomCtx()
  {
    int size = multi_objects.size();
    printf("size:%d\n", size);
    for (int i = 0; i < size; i++)
    {
      delete multi_objects[i];
      multi_objects[i] = NULL;
    }
    multi_objects.clear();
  };
};

/**
 * @brief Finds an object context with the specified object_id in the CustomCtx.
 *
 * This function searches for an object context within the CustomCtx structure
 * with a matching object_id. It uses a unique lock to ensure thread safety
 * while accessing the multi_objects vector.
 *
 * @param ctx A pointer to the CustomCtx structure containing object contexts.
 * @param object_id The unique identifier of the object context to be found.
 *
 * @return A pointer to the found SObjectContex if object_id matches, or NULL if not found.
 */
SObjectContex *
findObjectCtx(CustomCtx *ctx, guint64 object_id)
{
  // Acquire a unique lock on the mutex associated with the CustomCtx structure
  unique_lock<mutex> lck(ctx->mtx);

  // Declare a pointer to SObjectContex and initialize it to NULL
  SObjectContex *pSObjectCtx = NULL;

  // Iterate through the vector of SObjectContex pointers in CustomCtx
  for (ObjectIterator itor = ctx->multi_objects.begin(); itor != ctx->multi_objects.end(); itor++)
  {
    // Check if the object_id of the current SObjectContex matches the specified object_id
    if ((*itor)->object_id == object_id)
    {
      // If a match is found, assign the address of the matching SObjectContex to pSObjectCtx
      pSObjectCtx = (*itor);
    }
  }

  // Return the pointer to the found SObjectContex (or NULL if not found)
  return pSObjectCtx;
}

/**
 * @brief Finds an unused object context with the specified object_id in the CustomCtx.
 *
 * This function searches for an object context within the CustomCtx structure
 * with a matching object_id and an object_id equal to UNTRACKED_OBJECT_ID. It uses
 * a unique lock to ensure thread safety while accessing the multi_objects vector.
 *
 * @param ctx A pointer to the CustomCtx structure containing object contexts.
 * @param object_id The unique identifier of the object context to be found.
 *
 * @return A pointer to the found unused SObjectContex if an object with the specified
 *         object_id and UNTRACKED_OBJECT_ID is found, or NULL if not found.
 */
SObjectContex *
findUnusedObjectCtx(CustomCtx *ctx, guint64 object_id)
{
  // Acquire a unique lock on the mutex associated with the CustomCtx structure
  unique_lock<mutex> lck(ctx->mtx);

  // Declare a pointer to SObjectContex and initialize it to NULL
  SObjectContex *pSObjectCtx = NULL;

  // Iterate through the vector of SObjectContex pointers in CustomCtx
  for (ObjectIterator itor = ctx->multi_objects.begin(); itor != ctx->multi_objects.end(); itor++)
  {
    // Check if the object_id of the current SObjectContex is UNTRACKED_OBJECT_ID
    if ((*itor)->object_id == UNTRACKED_OBJECT_ID)
    {
      // If an unused object is found, assign the address of the matching SObjectContex to pSObjectCtx
      pSObjectCtx = (*itor);
    }
  }

  // Return the pointer to the found unused SObjectContex (or NULL if not found)
  return pSObjectCtx;
}

/**
 * @brief Creates and initializes a new object context within the CustomCtx.
 *
 * This function creates a new SObjectContex instance, allocates memory for its
 * keypoint arrays (x, y, z), and initializes other members such as object_id,
 * frameIndex, and adds it to the multi_objects vector in the CustomCtx. It uses
 * a unique lock to ensure thread safety during object creation and addition.
 *
 * @param ctx A pointer to the CustomCtx structure where the new object context will be added.
 *
 * @return A pointer to the newly created SObjectContex if successful, or NULL if memory allocation fails.
 */
SObjectContex *
CreateObjectCtx(CustomCtx *ctx)
{
  // Acquire a unique lock on the mutex associated with the CustomCtx structure
  unique_lock<mutex> lck(ctx->mtx);

  // Declare a pointer to SObjectContex and initialize it to a new instance
  SObjectContex *pSObjectCtx = new SObjectContex;

  // Check if the allocation was successful
  if (pSObjectCtx)
  {
    // Initialize object_id to UNTRACKED_OBJECT_ID
    pSObjectCtx->object_id = UNTRACKED_OBJECT_ID;

    // Allocate memory for x, y, and z arrays
    pSObjectCtx->x = (float *)calloc(ctx->one_channel_element_num, sizeof(float));
    pSObjectCtx->y = (float *)calloc(ctx->one_channel_element_num, sizeof(float));
    pSObjectCtx->z = (float *)calloc(ctx->one_channel_element_num, sizeof(float));

    // Initialize frameIndex to 0
    pSObjectCtx->frameIndex = 0;

    // Add the newly created object context to the multi_objects vector in CustomCtx
    ctx->multi_objects.push_back(pSObjectCtx);
  }

  // Return the pointer to the newly created SObjectContex (or NULL if allocation failed)
  return pSObjectCtx;
}

/**
 * @brief Resets the state of an object context within the CustomCtx.
 *
 * This function resets the specified SObjectContex by setting its object_id to
 * UNTRACKED_OBJECT_ID, clearing the keypoint arrays ('x', 'y', 'z'), resetting
 * frameIndex to 0, and timestamp (tv_sec) to 0. It ensures that the provided
 * object context pointer is not NULL before performing the reset.
 *
 * @param ctx A pointer to the CustomCtx structure containing the object context.
 * @param pSObjectCtx A pointer to the SObjectContex structure to be reset.
 */
void ResetObjectCtx(CustomCtx *ctx, SObjectContex *pSObjectCtx)
{
  // Check if the pointer to the object context is not NULL
  if (pSObjectCtx)
  {
    // Print a message indicating the start of the object context reset
    printf("ResetObjectCtx, object_id:%ld\n", pSObjectCtx->object_id);

    // Reset the object_id to UNTRACKED_OBJECT_ID
    pSObjectCtx->object_id = UNTRACKED_OBJECT_ID;

    // Set all elements in the 'x', 'y', and 'z' array to 0
    memset(pSObjectCtx->x, 0, ctx->one_channel_element_num * sizeof(float));
    memset(pSObjectCtx->y, 0, ctx->one_channel_element_num * sizeof(float));
    memset(pSObjectCtx->z, 0, ctx->one_channel_element_num * sizeof(float));

    // Reset the frameIndex to 0
    pSObjectCtx->frameIndex = 0;

    // Reset the tv_sec (timestamp) to 0
    pSObjectCtx->tv_sec = 0;
  }
}

/**
 * @brief Loop through object contexts in the custom context and reset those that exceed the timeout.
 *
 * Acquires a unique lock on the mutex associated with the custom context to ensure thread safety.
 * Iterates through the vector of object contexts, checks the timestamp of each tracked object, and
 * resets the object context if it exceeds the timeout threshold. Uses gettimeofday to obtain the
 * current time.
 *
 * @param ctx A pointer to the CustomCtx structure containing object contexts.
 */
void LoopObjectCtx(CustomCtx *ctx)
{
  // Acquire a unique lock on the mutex associated with the custom context
  unique_lock<mutex> lck(ctx->mtx);

  // Declare a timeval struct to store the current time
  struct timeval tv;

  // Declare a pointer to SObjectContex for iterating through the vector of object contexts
  SObjectContex *pSObjectCtx = NULL;

  // Iterate through the vector of object contexts in the custom context
  for (ObjectIterator itor = ctx->multi_objects.begin(); itor != ctx->multi_objects.end(); itor++)
  {
    // Obtain a pointer to the current object context
    pSObjectCtx = (*itor);

    // Get the current time using gettimeofday function
    gettimeofday(&tv, NULL);

    // Check if the object is tracked (object_id is not UNTRACKED_OBJECT_ID)
    // and if the time difference between current time and object's timestamp exceeds the timeout threshold
    if (pSObjectCtx->object_id != UNTRACKED_OBJECT_ID &&
        (pSObjectCtx->tv_sec - tv.tv_sec) > _TIME_OUT_)
    {
      // If the conditions are met, reset the object context using the ResetObjectCtx function
      ResetObjectCtx(ctx, pSObjectCtx);
    }
  }
}

/**
 * @brief Updates the keypoints data in the specified object context.
 *
 * This function is responsible for updating the keypoints data in the provided
 * SObjectContex structure. It moves the existing keypoints data, saves new
 * keypoints from the NvDsJoints structure, and updates the timestamp in the
 * object context. The function ensures that the provided object context pointer
 * is not NULL before performing the update.
 *
 * @param ctx A pointer to the CustomCtx structure containing the object context.
 * @param user_meta_data A pointer to user metadata, assumed to be of type NvDsJoints.
 * @param pSObjectCtx A pointer to the SObjectContex structure to be updated.
 */
void sveKeypoints(CustomCtx *ctx, void *user_meta_data, SObjectContex *pSObjectCtx)
{
  // Acquire a unique lock on the mutex associated with the custom context
  unique_lock<mutex> lck(ctx->mtx);

  // Check if the object context is not NULL
  if (pSObjectCtx)
  {
    // Cast the user metadata to NvDsJoints structure
    NvDsJoints *ds_joints = (NvDsJoints *)user_meta_data;

    // Move existing keypoints data from tail to head
    memmove(pSObjectCtx->x, pSObjectCtx->x + 34, ctx->move_element_num * sizeof(float));
    memmove(pSObjectCtx->y, pSObjectCtx->y + 34, ctx->move_element_num * sizeof(float));
    memmove(pSObjectCtx->z, pSObjectCtx->z + 34, ctx->move_element_num * sizeof(float));

    // Save new keypoints
    for (int i = 0; i < ds_joints->num_joints; i++)
    {
      // Update x, y, and z values in the object context
      *(pSObjectCtx->x + ctx->move_element_num + i) = ds_joints->joints[i].x;
      *(pSObjectCtx->y + ctx->move_element_num + i) = ds_joints->joints[i].y;
      *(pSObjectCtx->z + ctx->move_element_num + i) = ds_joints->joints[i].z;
    }

    // Update the timestamp in the object context
    struct timeval tv;
    gettimeofday(&tv, NULL);
    pSObjectCtx->tv_sec = tv.tv_sec;
  }
}

/**
 * @brief Prepares custom tensors for processing based on input metadata and keypoints.
 *
 * This function is responsible for preparing custom tensors for processing by acquiring
 * a buffer from the tensor pool, updating the buffer with keypoints data from the specified
 * object context, and handling object context extension or creation as needed. It iterates
 * through the input batch's frame and object metadata, searching for the object context
 * associated with the given object_id. If not found, it looks for an unused context or
 * extends the context and copies keypoints accordingly. The function copies keypoints to
 * the buffer in a specific format (3 X 300 X 34 X 1) and resets the object context if
 * a timeout is detected.
 *
 * @param ctx A pointer to the CustomCtx structure containing the object contexts.
 * @param batch A pointer to the NvDsPreProcessBatch structure containing input metadata.
 * @param buf A pointer to the NvDsPreProcessCustomBuf structure for acquiring the buffer.
 * @param tensorParam A reference to CustomTensorParams for additional tensor parameters.
 * @param acquirer A pointer to the NvDsPreProcessAcquirer for acquiring the buffer.
 *
 * @return The NvDsPreProcessStatus indicating the status of tensor preparation.
 */
NvDsPreProcessStatus
CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch,
                        NvDsPreProcessCustomBuf *&buf, CustomTensorParams &tensorParam,
                        NvDsPreProcessAcquirer *acquirer);
{
  // Initialize status variable
  NvDsPreProcessStatus status = NVDSPREPROCESS_TENSOR_NOT_READY;

  // Acquire a buffer from the tensor pool
  buf = acquirer->acquire();
  float *pDst = (float *)buf->memory_ptr; // Pointer to the memory of the acquired buffer
  int units = batch->units.size();        // Number of units in the batch

  // Iterate through units in the batch
  for (int i = 0; i < units; i++)
  {
    guint64 object_id = batch->units[i].roi_meta.object_meta->object_id; // Object ID
    GstBuffer *inbuf = (GstBuffer *)batch->inbuf;                        // Input buffer
    NvDsMetaList *l_frame = NULL;                                        // Metadata list for frames
    NvDsMetaList *l_obj = NULL;                                          // Metadata list for objects
    NvDsMetaList *l_user = NULL;                                         // Metadata list for users
    SObjectContex *pSObjectCtx = NULL;                                   // Object context
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);   // Batch metadata

    // Iterate through frame metadata in the batch
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data); // Frame metadata

      // Iterate through object metadata in the frame
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
      {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data; // Object metadata

        // Check if the object_id matches the current object
        if (obj_meta->object_id != object_id)
          continue;

        // Iterate through user metadata attached to the object
        for (l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
        {
          NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data; // User metadata

          // Check if the user metadata type is NVDS_OBJ_META
          if (user_meta->base_meta.meta_type == NVDS_OBJ_META)
          {
            // Find the SObjectContext using the object_id
            pSObjectCtx = findObjectCtx(ctx, obj_meta->object_id);

            if (!pSObjectCtx)
            {
              // If the object_id is not found, find an unused SObjectContext
              pSObjectCtx = findUnusedObjectCtx(ctx, obj_meta->object_id);

              if (pSObjectCtx)
              {
                // If an unused context is found, copy keypoints to it
                pSObjectCtx->object_id = obj_meta->object_id;
                sveKeypoints(ctx, user_meta->user_meta_data, pSObjectCtx);
              }
              else
              {
                // If no unused context is found, extend the context and copy keypoints
                pSObjectCtx = CreateObjectCtx(ctx);
                printf("extendObjectCtx pSObjectCtx:%p\n", pSObjectCtx);

                if (pSObjectCtx)
                {
                  pSObjectCtx->object_id = obj_meta->object_id;
                  sveKeypoints(ctx, user_meta->user_meta_data, pSObjectCtx);
                }
              }
            }
            else
            {
              // If the context is found, copy keypoints
              sveKeypoints(ctx, user_meta->user_meta_data, pSObjectCtx);
            }
          }
        }
      }
    }

    // Copy keypoints to the buffer in the specified format (3 X 300 X 34 X 1)
    if (pSObjectCtx)
    {
      int bufLen = ctx->one_channel_element_num * sizeof(float);
      cudaMemcpy(pDst, pSObjectCtx->x, bufLen, cudaMemcpyHostToDevice);
      cudaMemcpy(pDst + ctx->one_channel_element_num, pSObjectCtx->y, bufLen, cudaMemcpyHostToDevice);
      cudaMemcpy(pDst + ctx->two_channel_element_num, pSObjectCtx->z, bufLen, cudaMemcpyHostToDevice);
      pDst = pDst + 3 * bufLen;
    }
  }

  // Reset object context if timeout
  LoopObjectCtx(ctx);
  status = NVDSPREPROCESS_SUCCESS; // Update status
  return status;                   // Return the final status
}

/**
 * @brief Custom transformation function for pre-processing.
 *
 * This function performs a custom transformation on the input surface but, in this specific
 * implementation, does nothing. The body pose data is assumed to be present in the object's
 * metadata, and since the object metadata cannot be accessed in this context, no transformation
 * is applied. The function returns NVDSPREPROCESS_SUCCESS to indicate successful completion.
 *
 * @param in_surf A pointer to the input NvBufSurface.
 * @param out_surf A pointer to the output NvBufSurface.
 * @param params A reference to CustomTransformParams for additional transformation parameters.
 *
 * @return The NvDsPreProcessStatus indicating the success of the custom transformation.
 */
NvDsPreProcessStatus
CustomTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf,
                     CustomTransformParams &params);
{
  /* do nothing, bodypose data is in object's metadata, here we can't access object */
  return NVDSPREPROCESS_SUCCESS;
}

/**
 * @brief Initializes the custom library context.
 *
 * This function allocates memory for a new customctx object, retrieves the frames sequence length
 * from user configurations, validates the length, calculates the number of elements for different
 * channels, and initializes a vector for multi_keypoints. The created customctx object is returned.
 *
 * @param initparams A structure containing initialization parameters, including user configurations.
 *
 * @return A pointer to the initialized customctx object.
 */
customctx *initlib(custominitparams initparams)
{
  // Allocate memory for a new customctx object
  customctx *ctx = new customctx;

  // Retrieve the frames sequence length from user configurations
  std::string sframeseqlen = initparams.user_configs[nvdspreprocess_user_configs_frames_sequence_lenghth];

  // Convert the frames sequence length from string to integer
  int len = atoi(sframeseqlen.c_str());

  // Print the frames sequence length to the console
  printf("frameseqlen:%d\n", len);

  // Check if the frames sequence length is within a valid range
  if (len < _MIN_FRAME_ || len > _MAX_FRAME_)
  {
    // If out of range, set it to the maximum allowed value
    printf("frameseqlen illegal, use default value 300\n");
    len = _MAX_FRAME_;
  }

  // Calculate the number of elements for different channels
  ctx->one_channel_element_num = len * 34;
  ctx->two_channel_element_num = 2 * len * 34;
  ctx->move_element_num = (len - 1) * 34;

  // Initialize vector for multi_keypoints by calling createobjectctx function
  for (int i = 0; i < _MAX_OBJECT_NUM_; i++)
  {
    createobjectctx(ctx);
  }

  // Return the pointer to the initialized customctx object
  return ctx;
}

/**
 * @brief Deinitializes the custom library context.
 *
 * This function deallocates the memory occupied by the customctx object, freeing up resources.
 *
 * @param ctx A pointer to the customctx object to be deinitialized.
 */
void deinitlib(customctx *ctx)
{
  // Deallocate the memory occupied by the customctx object
  delete ctx;
}
