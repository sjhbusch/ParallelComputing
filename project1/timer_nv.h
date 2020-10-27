/*
 * Copyright 2012 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Usage example
 *
 * StartTimer();
 *
 * ...code section to time
 *
 * double elapsedTime = GetTimer(); //elapsed time is in seconds
 *
 *
 */
#ifndef TIMER_H
#define TIMER_H
#include <sys/time.h>
struct timeval timerStart;
void           StartTimer( ) { gettimeofday(&timerStart, NULL); }
// time elapsed in s
double GetTimer( )
{
    struct timeval timerStop;
    gettimeofday(&timerStop, NULL);
    return ((timerStop.tv_sec - timerStart.tv_sec)
            + (timerStop.tv_usec - timerStart.tv_usec) / 1000000.0);
}
#endif // TIMER_H
