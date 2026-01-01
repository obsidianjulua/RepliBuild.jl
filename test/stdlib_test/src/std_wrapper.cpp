#include "std_wrapper.h"
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>

// ============================================================================
// STRING HANDLING
// ============================================================================

extern "C" {

StringWrapper string_create(const char* initial_str) {
    StringWrapper s;
    if (initial_str) {
        s.length = strlen(initial_str);
        s.capacity = s.length + 1;
        s.data = (char*)malloc(s.capacity);
        strcpy(s.data, initial_str);
    } else {
        s.length = 0;
        s.capacity = 1;
        s.data = (char*)malloc(1);
        s.data[0] = '\0';
    }
    s.owns_data = true;
    return s;
}

void string_destroy(StringWrapper* s) {
    if (s && s->owns_data && s->data) {
        free(s->data);
        s->data = nullptr;
        s->length = 0;
        s->capacity = 0;
    }
}

const char* string_get(const StringWrapper* s) {
    return s ? s->data : nullptr;
}

void string_append(StringWrapper* dest, const char* suffix) {
    if (!dest || !suffix) return;
    
    size_t suffix_len = strlen(suffix);
    size_t new_len = dest->length + suffix_len;
    
    if (new_len + 1 > dest->capacity) {
        size_t new_cap = new_len + 1;
        // Simple growth strategy
        if (new_cap < dest->capacity * 2) new_cap = dest->capacity * 2;
        
        char* new_data = (char*)realloc(dest->data, new_cap);
        if (!new_data) return; // Allocation failed
        
        dest->data = new_data;
        dest->capacity = new_cap;
    }
    
    strcat(dest->data, suffix);
    dest->length = new_len;
}

void string_concat(StringWrapper* dest, const StringWrapper* src) {
    if (src && src->data) {
        string_append(dest, src->data);
    }
}

StringWrapper string_duplicate(const StringWrapper* src) {
    return string_create(src->data);
}

int string_compare(const StringWrapper* s1, const StringWrapper* s2) {
    if (!s1->data && !s2->data) return 0;
    if (!s1->data) return -1;
    if (!s2->data) return 1;
    return strcmp(s1->data, s2->data);
}

}

// ============================================================================
// FILE I/O WRAPPER
// ============================================================================

extern "C" {

FileHandle* file_open(const char* path, const char* mode) {
    FILE* f = fopen(path, mode);
    if (!f) return nullptr;

    FileHandle* fh = (FileHandle*)malloc(sizeof(FileHandle));
    fh->native_handle = f;
    strncpy(fh->mode, mode, 3);
    fh->mode[3] = '\0';
    fh->is_open = true;
    fh->last_error = 0;
    return fh;
}

void file_close(FileHandle* handle) {
    if (handle) {
        if (handle->is_open && handle->native_handle) {
            fclose(handle->native_handle);
            handle->native_handle = nullptr;
            handle->is_open = false;
        }
        free(handle);
    }
}

size_t file_write(FileHandle* handle, const char* data, size_t size) {
    if (!handle || !handle->is_open) return 0;
    return fwrite(data, 1, size, handle->native_handle);
}

size_t file_read(FileHandle* handle, char* buffer, size_t size) {
    if (!handle || !handle->is_open) return 0;
    return fread(buffer, 1, size, handle->native_handle);
}

int file_flush(FileHandle* handle) {
    if (!handle || !handle->is_open) return EOF;
    return fflush(handle->native_handle);
}

long file_tell(FileHandle* handle) {
    if (!handle || !handle->is_open) return -1L;
    return ftell(handle->native_handle);
}

int file_seek(FileHandle* handle, long offset, int origin) {
    if (!handle || !handle->is_open) return -1;
    return fseek(handle->native_handle, offset, origin);
}

}

// ============================================================================
// TIME OPERATIONS
// ============================================================================

extern "C" {

DateInfo time_get_current_utc() {
    DateInfo info = {0};
    time_t rawtime;
    struct tm* ptm;
    
    time(&rawtime);
    ptm = gmtime(&rawtime);
    
    if (ptm) {
        info.year = ptm->tm_year + 1900;
        info.month = ptm->tm_mon + 1;
        info.day = ptm->tm_mday;
        info.hour = ptm->tm_hour;
        info.minute = ptm->tm_min;
        info.second = ptm->tm_sec;
    }
    
    // Get nanoseconds (platform dependent, using chrono for portable C++)
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    info.nanoseconds = nanos % 1000000000;
    
    return info;
}

DateInfo time_get_current_local() {
    DateInfo info = {0};
    time_t rawtime;
    struct tm* ptm;
    
    time(&rawtime);
    ptm = localtime(&rawtime);
    
    if (ptm) {
        info.year = ptm->tm_year + 1900;
        info.month = ptm->tm_mon + 1;
        info.day = ptm->tm_mday;
        info.hour = ptm->tm_hour;
        info.minute = ptm->tm_min;
        info.second = ptm->tm_sec;
    }
    
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    info.nanoseconds = nanos % 1000000000;
    
    return info;
}

double time_diff_seconds(DateInfo start, DateInfo end) {
    struct tm tm_start = {0};
    tm_start.tm_year = start.year - 1900;
    tm_start.tm_mon = start.month - 1;
    tm_start.tm_mday = start.day;
    tm_start.tm_hour = start.hour;
    tm_start.tm_min = start.minute;
    tm_start.tm_sec = start.second;
    
    struct tm tm_end = {0};
    tm_end.tm_year = end.year - 1900;
    tm_end.tm_mon = end.month - 1;
    tm_end.tm_mday = end.day;
    tm_end.tm_hour = end.hour;
    tm_end.tm_min = end.minute;
    tm_end.tm_sec = end.second;
    
    time_t t_start = mktime(&tm_start);
    time_t t_end = mktime(&tm_end);
    
    double diff = difftime(t_end, t_start);
    diff += (end.nanoseconds - start.nanoseconds) / 1e9;
    
    return diff;
}

void time_sleep_ms(unsigned int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

}

// ============================================================================
// RECURSIVE DATA STRUCTURES (Linked List)
// ============================================================================

extern "C" {

LinkedList list_create() {
    LinkedList list;
    list.head = nullptr;
    list.tail = nullptr;
    list.size = 0;
    return list;
}

void list_destroy(LinkedList* list) {
    if (!list) return;
    list_clear(list);
    // Note: LinkedList struct itself is usually stack allocated or managed by caller
    // based on list_create returning value. 
}

void list_push_back(LinkedList* list, int value) {
    if (!list) return;
    
    ListNode* node = (ListNode*)malloc(sizeof(ListNode));
    node->value = value;
    node->next = nullptr;
    node->prev = list->tail;
    
    if (list->tail) {
        list->tail->next = node;
    } else {
        list->head = node;
    }
    list->tail = node;
    list->size++;
}

void list_push_front(LinkedList* list, int value) {
    if (!list) return;
    
    ListNode* node = (ListNode*)malloc(sizeof(ListNode));
    node->value = value;
    node->next = list->head;
    node->prev = nullptr;
    
    if (list->head) {
        list->head->prev = node;
    } else {
        list->tail = node;
    }
    list->head = node;
    list->size++;
}

int list_pop_back(LinkedList* list) {
    if (!list || !list->tail) return 0; // Should handle error better
    
    ListNode* node = list->tail;
    int value = node->value;
    
    list->tail = node->prev;
    if (list->tail) {
        list->tail->next = nullptr;
    } else {
        list->head = nullptr;
    }
    
    free(node);
    list->size--;
    return value;
}

int list_pop_front(LinkedList* list) {
    if (!list || !list->head) return 0;
    
    ListNode* node = list->head;
    int value = node->value;
    
    list->head = node->next;
    if (list->head) {
        list->head->prev = nullptr;
    } else {
        list->tail = nullptr;
    }
    
    free(node);
    list->size--;
    return value;
}

ListNode* list_find(LinkedList* list, int value) {
    if (!list) return nullptr;
    
    ListNode* current = list->head;
    while (current) {
        if (current->value == value) return current;
        current = current->next;
    }
    return nullptr;
}

void list_clear(LinkedList* list) {
    if (!list) return;
    
    ListNode* current = list->head;
    while (current) {
        ListNode* next = current->next;
        free(current);
        current = next;
    }
    list->head = nullptr;
    list->tail = nullptr;
    list->size = 0;
}

}
