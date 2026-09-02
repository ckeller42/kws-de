cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
# Pinned in firmware/README.md and .github/workflows/firmware.yml too — keep all three equal.
set(KWS_DE_IDF_PIN "v5.5.5")
if(NOT "$ENV{IDF_VERSION}" STREQUAL "" AND NOT "$ENV{IDF_VERSION}" MATCHES "^${KWS_DE_IDF_PIN}")
  message(WARNING "kws-de firmware is pinned to ESP-IDF ${KWS_DE_IDF_PIN}, building with $ENV{IDF_VERSION}")
endif()
project(kws_de_fw)
