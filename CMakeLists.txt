# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
# Рекомендуемая CLI-сборка (артефакты в ransform/out; нужен Developer Command Prompt для cl.exe):
#   cmake --preset x64-release
#   cmake --build ransform/out/build/x64-release
#   ctest --test-dir ransform/out/build/x64-release
# Либо без пресетов (Ninja + MSVC):
#   cmake -S . -B ransform/out/build/ninja-release -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build ransform/out/build/ninja-release
#   ctest --test-dir ransform/out/build/ninja-release
#
cmake_minimum_required (VERSION 3.20)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("ransform")

# Включите подпроекты.
add_subdirectory ("ransform")
