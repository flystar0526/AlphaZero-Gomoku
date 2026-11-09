# Invalid arguments must fail before initializing either the model or the GUI.
function(check_invalid)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=DISPLAY --unset=WAYLAND_DISPLAY
      "${GOMOKU_EXECUTABLE}" ${ARGV}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if (NOT result STREQUAL "1" OR NOT error MATCHES "Usage: gomoku")
    message(FATAL_ERROR "Unexpected CLI result for '${ARGV}': ${result}\n${output}${error}")
  endif()
endfunction()

check_invalid(--simulation)
check_invalid(--simulation 0)
check_invalid(--simulation -1)
check_invalid(--simulation 1.5)
check_invalid(--simulation 3junk)
check_invalid(--simulation nan)
check_invalid(--simulation 2147483648)
check_invalid(--simulation 9999999999999999999999999)
check_invalid(--simulation 2 extra)
check_invalid(--unknown 2)
check_invalid(--simulation 2 --simulation 2)

execute_process(
  COMMAND "${GOMOKU_EXECUTABLE}" --help
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
)
if (NOT result STREQUAL "0" OR NOT output MATCHES "Usage: gomoku")
  message(FATAL_ERROR "--help failed: ${result}\n${output}")
endif()
