cmake_minimum_required(VERSION 3.10)

project(3DObjectViewer)

set(CMAKE_CXX_STANDARD 17)

# Пути к библиотекам
set(GLFW_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/glfw/include")
set(GLEW_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/glew/include")
set(GLM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/glm")
set(EIGEN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/eigen")
set(IMGUI_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/imgui")
set(IMGUI_FILE_DIALOG_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/ImGuiFileDialog")

# Подключение директорий с заголовочными файлами
include_directories(${GLFW_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${GLM_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR} ${IMGUI_INCLUDE_DIR} ${IMGUI_FILE_DIALOG_INCLUDE_DIR})

# Путь к библиотекам
link_directories(${CMAKE_SOURCE_DIR}/libs/glfw/lib-vc2022)
link_directories(${CMAKE_SOURCE_DIR}/libs/glew/lib/Release/x64)

# Импортируем исходные файлы ImGui и ImGuiFileDialog
file(GLOB IMGUI_SRC
    "${CMAKE_SOURCE_DIR}/libs/imgui/*.cpp"
    "${CMAKE_SOURCE_DIR}/libs/imgui/backends/imgui_impl_glfw.cpp"
    "${CMAKE_SOURCE_DIR}/libs/imgui/backends/imgui_impl_opengl3.cpp"
)

file(GLOB IMGUI_FILE_DIALOG_SRC
    "${CMAKE_SOURCE_DIR}/libs/ImGuiFileDialog/ImGuiFileDialog.cpp"
)

# Создаем статические библиотеки для ImGui и ImGuiFileDialog
add_library(imgui STATIC ${IMGUI_SRC})
add_library(ImGuiFileDialog STATIC ${IMGUI_FILE_DIALOG_SRC})

# Добавляем главный исполняемый файл
add_executable(3DObjectViewer "${CMAKE_SOURCE_DIR}/main.cpp")

# Линковка библиотек (используем glew32s для статической линковки)
target_link_libraries(3DObjectViewer glfw3 glew32s imgui ImGuiFileDialog opengl32)

# Для ImGuiFileDialog
target_include_directories(3DObjectViewer PRIVATE ${IMGUI_FILE_DIALOG_INCLUDE_DIR})
