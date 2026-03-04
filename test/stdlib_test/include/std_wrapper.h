#pragma once

#include <cstddef>
#include <cstdio>
#include <ctime>
#include <cstdint>

// ============================================================================
// STDLIB TEST: Standard C Library Wrappers for Julia FFI
// ============================================================================

// ============================================================================
// STRING HANDLING
// ============================================================================

struct StringWrapper {
    char* data;
    size_t length;
    size_t capacity;
    bool owns_data;
};

extern "C" {

StringWrapper string_create(const char* initial_str);
void string_destroy(StringWrapper* s);
const char* string_get(const StringWrapper* s);
void string_append(StringWrapper* dest, const char* suffix);
void string_concat(StringWrapper* dest, const StringWrapper* src);
StringWrapper string_duplicate(const StringWrapper* src);
int string_compare(const StringWrapper* s1, const StringWrapper* s2);

}

// ============================================================================
// FILE I/O WRAPPER
// ============================================================================

struct FileHandle {
    FILE* native_handle;
    char mode[4];
    bool is_open;
    int last_error;
};

extern "C" {

FileHandle* file_open(const char* path, const char* mode);
void file_close(FileHandle* handle);
size_t file_write(FileHandle* handle, const char* data, size_t size);
size_t file_read(FileHandle* handle, char* buffer, size_t size);
int file_flush(FileHandle* handle);
long file_tell(FileHandle* handle);
int file_seek(FileHandle* handle, long offset, int origin);

}

// ============================================================================
// TIME OPERATIONS
// ============================================================================

struct DateInfo {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
    long nanoseconds;
};

extern "C" {

DateInfo time_get_current_utc();
DateInfo time_get_current_local();
double time_diff_seconds(DateInfo start, DateInfo end);
void time_sleep_ms(unsigned int milliseconds);

}

// ============================================================================
// RECURSIVE DATA STRUCTURES (Linked List)
// ============================================================================

struct ListNode {
    int value;
    ListNode* next;
    ListNode* prev;
};

struct LinkedList {
    ListNode* head;
    ListNode* tail;
    size_t size;
};

extern "C" {

LinkedList list_create();
void list_destroy(LinkedList* list);
void list_push_back(LinkedList* list, int value);
void list_push_front(LinkedList* list, int value);
int list_pop_back(LinkedList* list);
int list_pop_front(LinkedList* list);
ListNode* list_find(LinkedList* list, int value);
void list_clear(LinkedList* list);

}
