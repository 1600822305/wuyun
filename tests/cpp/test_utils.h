#pragma once
/**
 * 测试工具头文件 — Windows控制台UTF-8输出修复
 * 在 main() 开头调用 init_test_console() 即可
 */

#ifdef _WIN32
#include <windows.h>
#endif

inline void init_test_console() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);  // UTF-8
    SetConsoleCP(65001);
#endif
}
