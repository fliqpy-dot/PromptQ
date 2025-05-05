# queue_system.py
import time
import threading
from collections import deque
import random
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Union


class SimpleQueue:
    """A basic FIFO (First-In-First-Out) queue implementation."""
    
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add an item to the end of the queue."""
        self.items.append(item)
        return True
    
    def dequeue(self):
        """Remove and return the item at the front of the queue."""
        if self.is_empty():
            return None
        return self.items.popleft()
    
    def peek(self):
        """Return the item at the front of the queue without removing it."""
        if self.is_empty():
            return None
        return self.items[0]
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self.items)
    
    def clear(self):
        """Remove all items from the queue."""
        self.items.clear()
    
    def display(self):
        """Return the current state of the queue as a list."""
        return list(self.items)
    
    def __str__(self):
        return f"Queue({list(self.items)})"


class CircularQueue:
    """A fixed-size circular queue implementation."""
    
    def __init__(self, capacity):
        """Initialize a circular queue with given capacity."""
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        """Add an item to the queue if space is available."""
        if self.is_full():
            return False
        
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1
        return True
    
    def dequeue(self):
        """Remove and return the item at the front of the queue."""
        if self.is_empty():
            return None
        
        item = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self):
        """Return the item at the front of the queue without removing it."""
        if self.is_empty():
            return None
        return self.queue[self.front]
    
    def is_empty(self):
        """Check if the queue is empty."""
        return self.size == 0
    
    def is_full(self):
        """Check if the queue is full."""
        return self.size == self.capacity
    
    def get_size(self):
        """Return the current number of items in the queue."""
        return self.size
    
    def clear(self):
        """Remove all items from the queue."""
        self.queue = [None] * self.capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def display(self):
        """Return the current state of the queue as a list."""
        if self.is_empty():
            return []
        
        result = []
        index = self.front
        for _ in range(self.size):
            result.append(self.queue[index])
            index = (index + 1) % self.capacity
        
        return result
    
    def __str__(self):
        if self.is_empty():
            return "CircularQueue([])"
        
        result = []
        index = self.front
        for _ in range(self.size):
            result.append(self.queue[index])
            index = (index + 1) % self.capacity
        
        return f"CircularQueue({result})"


@dataclass(order=True)
class PriorityItem:
    """Wrapper class for items in the priority queue."""
    priority: int
    item: Any = field(compare=False)
    insertion_order: int = field(compare=True)


class PriorityQueue:
    """A priority queue implementation."""
    
    def __init__(self, is_min_heap=True):
        """
        Initialize a priority queue.
        
        Args:
            is_min_heap: If True, lower priority values are dequeued first.
                         If False, higher priority values are dequeued first.
        """
        self._queue = []
        self._index = 0  # Used to maintain FIFO order for items with the same priority
        self._is_min_heap = is_min_heap
    
    def enqueue(self, item, priority=0):
        """
        Add an item to the queue with the specified priority.
        
        Args:
            item: The item to be added to the queue.
            priority: The priority of the item. Lower values mean higher priority
                     if is_min_heap is True, otherwise higher values mean higher priority.
        """
        # Handle the case where item is a dict with "value" and "priority" keys
        if isinstance(item, dict) and "value" in item and "priority" in item:
            priority = item["priority"]
            item = item["value"]
        
        # Adjust priority for max-heap behavior if needed
        actual_priority = priority if self._is_min_heap else -priority
        heapq.heappush(self._queue, PriorityItem(actual_priority, item, self._index))
        self._index += 1
        return True
    
    def dequeue(self):
        """Remove and return the highest priority item from the queue."""
        if self.is_empty():
            return None
        return heapq.heappop(self._queue).item
    
    def peek(self):
        """Return the highest priority item without removing it."""
        if self.is_empty():
            return None
        return self._queue[0].item
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self._queue) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self._queue)
    
    def clear(self):
        """Remove all items from the queue."""
        self._queue = []
        self._index = 0
    
    def display(self):
        """Return the current state of the queue as a list of (priority, item) tuples."""
        return [(item.priority if self._is_min_heap else -item.priority, item.item) 
                for item in sorted(self._queue)]
    
    def __str__(self):
        type_name = "Min" if self._is_min_heap else "Max"
        items = [(item.priority if self._is_min_heap else -item.priority, item.item) 
                 for item in sorted(self._queue)]
        return f"{type_name}PriorityQueue({items})"


class ThreadSafeQueue:
    """A thread-safe queue implementation using locks."""
    
    def __init__(self):
        self.queue = deque()
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
    
    def enqueue(self, item):
        """Add an item to the queue."""
        with self.lock:
            self.queue.append(item)
            self.not_empty.notify()
        return True
    
    def dequeue(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.
        
        Args:
            block: If True and the queue is empty, block until an item is available.
            timeout: If block is True, block for at most timeout seconds.
                     If None, block indefinitely.
        
        Returns:
            An item from the queue.
        
        Raises:
            IndexError: If the queue is empty and block is False.
            TimeoutError: If the timeout is reached.
        """
        with self.lock:
            if self.is_empty() and not block:
                raise IndexError("Queue is empty")
            
            if self.is_empty() and block:
                # Wait until an item is available or timeout
                if not self.not_empty.wait(timeout=timeout):
                    raise TimeoutError("Timed out waiting for item")
            
            return self.queue.popleft()
    
    def is_empty(self):
        """Check if the queue is empty."""
        with self.lock:
            return len(self.queue) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        with self.lock:
            return len(self.queue)
    
    def clear(self):
        """Remove all items from the queue."""
        with self.lock:
            self.queue.clear()
    
    def display(self):
        """Return the current state of the queue as a list."""
        with self.lock:
            return list(self.queue)
    
    def __str__(self):
        with self.lock:
            return f"ThreadSafeQueue({list(self.queue)})"


class TaskQueue:
    """A queue for managing task execution."""
    
    def __init__(self, num_workers=1):
        """
        Initialize a task queue with the specified number of worker threads.
        
        Args:
            num_workers: The number of worker threads to create.
        """
        self.tasks = ThreadSafeQueue()
        self.results = {}
        self.results_lock = threading.Lock()
        self.workers = []
        self.running = False
        self.task_id_counter = 0
        self.task_id_lock = threading.Lock()
        
        # Start worker threads
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            self.workers.append(worker)
    
    def start(self):
        """Start processing tasks."""
        if not self.running:
            self.running = True
            for worker in self.workers:
                worker.start()
    
    def stop(self):
        """Stop processing tasks.""" 
        self.running = False
    
    def add_task(self, func, *args, **kwargs):
        """
        Add a task to the queue.
        
        Args:
            func: The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        
        Returns:
            A task ID that can be used to retrieve the result.
        """
        with self.task_id_lock:
            task_id = self.task_id_counter
            self.task_id_counter += 1
        
        task = (task_id, func, args, kwargs)
        self.tasks.enqueue(task)
        return task_id
    
    def get_result(self, task_id, block=True, timeout=None):
        """
        Get the result of a task.
        
        Args:
            task_id: The ID of the task.
            block: If True, block until the result is available.
            timeout: If block is True, block for at most timeout seconds.
        
        Returns:
            The result of the task.
        
        Raises:
            KeyError: If the task ID is not found.
            TimeoutError: If the timeout is reached.
        """
        start_time = time.time()
        while block:
            with self.results_lock:
                if task_id in self.results:
                    return self.results[task_id]
            
            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for result of task {task_id}")
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.01)
        
        # Non-blocking case
        with self.results_lock:
            if task_id in self.results:
                return self.results[task_id]
            else:
                raise KeyError(f"No result available for task {task_id}")
    
    def _worker_loop(self):
        """Worker thread function."""
        while self.running:
            try:
                # Get a task from the queue with a timeout
                task = self.tasks.dequeue(block=True, timeout=0.5)
                task_id, func, args, kwargs = task
                
                # Execute the task
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False
                
                # Store the result
                with self.results_lock:
                    self.results[task_id] = (success, result)
                
            except TimeoutError:
                # No tasks available, continue waiting
                pass
            except Exception as e:
                print(f"Error in worker thread: {e}")


# Example usage demonstration
def queue_demo():
    """Demonstrate the usage of different queue implementations."""
    
    print("=== Simple Queue Demo ===")
    queue = SimpleQueue()
    print(f"Empty queue: {queue}")
    
    for i in range(1, 6):
        queue.enqueue(i)
        print(f"Enqueued {i}: {queue}")
    
    while not queue.is_empty():
        item = queue.dequeue()
        print(f"Dequeued {item}: {queue}")
    
    print("\n=== Circular Queue Demo ===")
    circular_queue = CircularQueue(3)
    print(f"Empty circular queue: {circular_queue}")
    
    for i in range(1, 4):
        circular_queue.enqueue(i)
        print(f"Enqueued {i}: {circular_queue}")
    
    print(f"Queue is full: {circular_queue.is_full()}")
    
    while not circular_queue.is_empty():
        item = circular_queue.dequeue()
        print(f"Dequeued {item}: {circular_queue}")
    
    print("\n=== Priority Queue Demo ===")
    # Min priority queue (lower priority values dequeued first)
    pq = PriorityQueue(is_min_heap=True)
    
    # Add items with priorities
    items = [("Task A", 3), ("Task B", 1), ("Task C", 2), ("Task D", 1)]
    for item, priority in items:
        pq.enqueue(item, priority)
        print(f"Enqueued {item} with priority {priority}")
    
    print(f"Priority Queue state: {pq}")
    
    # Dequeue items based on priority
    while not pq.is_empty():
        item = pq.dequeue()
        print(f"Dequeued {item}: {pq}")
    
    print("\n=== Thread-Safe Queue Demo ===")
    thread_safe_queue = ThreadSafeQueue()
    
    # Enqueue and dequeue in separate threads
    def producer():
        for i in range(1, 6):
            thread_safe_queue.enqueue(i)
            print(f"Enqueued {i}: {thread_safe_queue}")
            time.sleep(1)
    
    def consumer():
        for _ in range(5):
            item = thread_safe_queue.dequeue()
            print(f"Dequeued {item}: {thread_safe_queue}")
            time.sleep(2)
    
    threading.Thread(target=producer).start()
    threading.Thread(target=consumer).start()

# Uncomment to run the demo function to test the queues
# queue_demo()