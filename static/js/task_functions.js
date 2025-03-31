// Store the current task ID for use in event handlers
let currentTaskId = null;

// Function to load project tasks from the API
function loadProjectTasks() {
    const projectId = projectIdValue; // This should be set by the template
    const tasksLoading = document.getElementById('tasks-loading');
    const tasksError = document.getElementById('tasks-error');
    const kanbanBoard = document.getElementById('kanban-board');
    
    if (!tasksLoading || !tasksError || !kanbanBoard) {
        console.error('Required task elements not found');
        return;
    }
    
    // Show loading, hide error and board
    tasksLoading.style.display = 'block';
    tasksError.style.display = 'none';
    kanbanBoard.style.display = 'none';
    
    // Clear existing tasks
    document.getElementById('todo-tasks').innerHTML = '';
    document.getElementById('in-progress-tasks').innerHTML = '';
    document.getElementById('review-tasks').innerHTML = '';
    document.getElementById('done-tasks').innerHTML = '';
    
    // Reset counts
    document.getElementById('todo-count').innerText = '0';
    document.getElementById('in-progress-count').innerText = '0';
    document.getElementById('review-count').innerText = '0';
    document.getElementById('done-count').innerText = '0';
    
    // Load tasks from API
    fetch(`/api/projects/${projectId}/tasks`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load tasks');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading, show board
            tasksLoading.style.display = 'none';
            kanbanBoard.style.display = 'flex';
            
            // Sort tasks by their status
            const todoTasks = data.filter(task => task.status === 'todo');
            const inProgressTasks = data.filter(task => task.status === 'in_progress');
            const reviewTasks = data.filter(task => task.status === 'review');
            const doneTasks = data.filter(task => task.status === 'done');
            
            // Render tasks in their respective columns
            renderTasks(todoTasks, 'todo-tasks', 'todo-count');
            renderTasks(inProgressTasks, 'in-progress-tasks', 'in-progress-count');
            renderTasks(reviewTasks, 'review-tasks', 'review-count');
            renderTasks(doneTasks, 'done-tasks', 'done-count');
            
            // Add click event to all task cards
            document.querySelectorAll('.task-card').forEach(card => {
                card.addEventListener('click', function() {
                    const taskId = this.getAttribute('data-task-id');
                    openTaskDetail(taskId);
                });
            });
        })
        .catch(error => {
            console.error('Error loading tasks:', error);
            tasksLoading.style.display = 'none';
            tasksError.style.display = 'block';
            const errorMessage = document.getElementById('tasks-error-message');
            if (errorMessage) {
                errorMessage.innerText = error.message;
            }
        });
}

// Function to render tasks in a column
function renderTasks(tasks, containerId, countId) {
    const container = document.getElementById(containerId);
    const count = document.getElementById(countId);
    
    if (!container || !count) return;
    
    count.innerText = tasks.length;
    
    tasks.forEach(task => {
        const taskCard = document.createElement('div');
        taskCard.className = `task-card task-priority-${task.priority}`;
        taskCard.setAttribute('data-task-id', task._id);
        
        let assigneeInitial = '';
        if (task.assigned_to) {
            assigneeInitial = task.assigned_to.substring(0, 1).toUpperCase();
        }
        
        taskCard.innerHTML = `
            <div class="task-title">${task.title}</div>
            <div class="task-meta">
                ${task.assigned_to ? 
                `<div class="task-assignee">
                    <div class="task-assignee-avatar">${assigneeInitial}</div>
                    ${task.assigned_to}
                </div>` : 
                `<div class="task-assignee">לא הוקצה</div>`}
                <div class="task-date">${task.due_date ? new Date(task.due_date).toLocaleDateString() : ''}</div>
            </div>
        `;
        
        container.appendChild(taskCard);
    });
}

// Function to open task detail
function openTaskDetail(taskId) {
    currentTaskId = taskId; // Store the task ID for use in event handlers
    
    const detailModal = new bootstrap.Modal(document.getElementById('taskDetailModal'));
    const loading = document.getElementById('taskDetailLoading');
    const content = document.getElementById('taskDetailContent');
    
    if (!detailModal || !loading || !content) {
        console.error('Task detail modal elements not found');
        return;
    }
    
    // Show loading, hide content
    loading.style.display = 'block';
    content.style.display = 'none';
    
    // Open modal
    detailModal.show();
    
    // Fetch task details
    fetch(`/api/tasks/${taskId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load task details');
            }
            return response.json();
        })
        .then(task => {
            // Update task details in the modal
            const titleElement = document.getElementById('taskDetailTitleText');
            const descriptionElement = document.getElementById('taskDetailDescription');
            const statusElement = document.getElementById('taskDetailStatus');
            const priorityElement = document.getElementById('taskDetailPriority');
            const dueDateElement = document.getElementById('taskDetailDueDate');
            const assigneeElement = document.getElementById('taskDetailAssignee');
            const createdByElement = document.getElementById('taskDetailCreatedBy');
            
            if (titleElement) titleElement.textContent = task.title;
            if (descriptionElement) descriptionElement.textContent = task.description || 'אין תיאור';
            
            // Format task status
            let statusText = '';
            switch(task.status) {
                case 'todo': statusText = 'לביצוע'; break;
                case 'in_progress': statusText = 'בתהליך'; break;
                case 'review': statusText = 'בבדיקה'; break;
                case 'done': statusText = 'הושלם'; break;
                default: statusText = task.status;
            }
            if (statusElement) statusElement.textContent = statusText;
            
            // Format task priority
            let priorityText = '';
            switch(task.priority) {
                case 'low': priorityText = 'נמוכה'; break;
                case 'medium': priorityText = 'בינונית'; break;
                case 'high': priorityText = 'גבוהה'; break;
                case 'urgent': priorityText = 'דחופה'; break;
                default: priorityText = task.priority;
            }
            if (priorityElement) priorityElement.textContent = priorityText;
            
            // Format due date
            const dueDate = task.due_date ? new Date(task.due_date).toLocaleDateString() : 'לא נקבע';
            if (dueDateElement) dueDateElement.textContent = dueDate;
            
            // Assignee and creator
            if (assigneeElement) assigneeElement.textContent = task.assigned_to || 'לא הוקצה';
            if (createdByElement) createdByElement.textContent = task.created_by || 'לא ידוע';
            
            // Load comments
            const commentsContainer = document.getElementById('taskComments');
            if (commentsContainer) {
                commentsContainer.innerHTML = '';
                
                if (task.comments && task.comments.length > 0) {
                    task.comments.forEach(comment => {
                        const commentElement = document.createElement('div');
                        commentElement.className = 'comment-item';
                        
                        const initial = comment.user.substring(0, 1).toUpperCase();
                        const date = new Date(comment.timestamp).toLocaleString();
                        
                        commentElement.innerHTML = `
                            <div class="comment-avatar">${initial}</div>
                            <div class="comment-content">
                                <div class="comment-header">
                                    <span class="comment-author">${comment.user}</span>
                                    <span class="comment-date">${date}</span>
                                </div>
                                <div class="comment-text">${comment.text}</div>
                            </div>
                        `;
                        
                        commentsContainer.appendChild(commentElement);
                    });
                } else {
                    commentsContainer.innerHTML = '<div class="text-center text-muted py-3">אין תגובות עדיין</div>';
                }
            }
            
            // Setup event listeners for task actions
            setupTaskDetailEventListeners(task);
            
            // Hide loading, show content
            loading.style.display = 'none';
            content.style.display = 'block';
        })
        .catch(error => {
            console.error('Error loading task details:', error);
            alert('שגיאה בטעינת פרטי המשימה: ' + error.message);
            detailModal.hide();
        });
}

// Setup event listeners for task detail buttons
function setupTaskDetailEventListeners(task) {
    // Edit button
    const editBtn = document.getElementById('editTaskBtn');
    if (editBtn) {
        // Remove previous event listeners by cloning
        const newEditBtn = editBtn.cloneNode(true);
        editBtn.parentNode.replaceChild(newEditBtn, editBtn);
        
        newEditBtn.addEventListener('click', function() {
            editTask(task);
        });
    }
    
    // Delete button
    const deleteBtn = document.getElementById('deleteTaskBtn');
    if (deleteBtn) {
        // Remove previous event listeners by cloning
        const newDeleteBtn = deleteBtn.cloneNode(true);
        deleteBtn.parentNode.replaceChild(newDeleteBtn, deleteBtn);
        
        newDeleteBtn.addEventListener('click', function() {
            deleteTask(task._id);
        });
    }
    
    // Add comment button
    const addCommentBtn = document.getElementById('addCommentBtn');
    if (addCommentBtn) {
        // Remove previous event listeners by cloning
        const newAddCommentBtn = addCommentBtn.cloneNode(true);
        addCommentBtn.parentNode.replaceChild(newAddCommentBtn, addCommentBtn);
        
        newAddCommentBtn.addEventListener('click', function() {
            addComment(task._id);
        });
    }
}

// Function to create a new task
function createTask() {
    const titleElement = document.getElementById('taskTitle');
    const descriptionElement = document.getElementById('taskDescription');
    const assigneeElement = document.getElementById('taskAssignee');
    const dueDateElement = document.getElementById('taskDueDate');
    const priorityElement = document.getElementById('taskPriority');
    
    if (!titleElement) {
        console.error('Task title element not found');
        return;
    }
    
    const title = titleElement.value;
    const description = descriptionElement ? descriptionElement.value : '';
    const assignee = assigneeElement ? assigneeElement.value : '';
    const dueDate = dueDateElement ? dueDateElement.value : '';
    const priority = priorityElement ? priorityElement.value : 'medium';
    
    if (!title) {
        alert('כותרת המשימה חובה');
        return;
    }
    
    const taskData = {
        title: title,
        description: description,
        assigned_to: assignee,
        due_date: dueDate,
        priority: priority,
        project_id: projectIdValue // This should be set by the template
    };
    
    fetch(`/api/projects/${projectIdValue}/tasks`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('שגיאה ביצירת משימה');
        }
        return response.json();
    })
    .then(data => {
        // Close modal and reload tasks
        const modal = bootstrap.Modal.getInstance(document.getElementById('addTaskModal'));
        if (modal) modal.hide();
        
        // Clear form
        const form = document.getElementById('addTaskForm');
        if (form) form.reset();
        
        // Reload tasks
        loadProjectTasks();
    })
    .catch(error => {
        console.error('Error creating task:', error);
        alert('שגיאה ביצירת המשימה: ' + error.message);
    });
}

// Function to add comment to task
function addComment(taskId) {
    const commentTextElement = document.getElementById('newComment');
    if (!commentTextElement) {
        console.error('Comment textarea not found');
        return;
    }
    
    const commentText = commentTextElement.value;
    
    if (!commentText.trim()) {
        alert('אנא הכנס טקסט לתגובה');
        return;
    }
    
    fetch(`/api/tasks/${taskId}/comments`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: commentText })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('שגיאה בהוספת התגובה');
        }
        return response.json();
    })
    .then(data => {
        // Clear comment input
        commentTextElement.value = '';
        
        // Reload task details to show new comment
        openTaskDetail(taskId);
    })
    .catch(error => {
        console.error('Error adding comment:', error);
        alert('שגיאה בהוספת התגובה: ' + error.message);
    });
}

// Function to delete task
function deleteTask(taskId) {
    if (!confirm('האם אתה בטוח שברצונך למחוק את המשימה הזו?')) {
        return;
    }
    
    fetch(`/api/tasks/${taskId}`, {
        method: 'DELETE'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('שגיאה במחיקת המשימה');
        }
        return response.json();
    })
    .then(data => {
        // Close modal
        const detailModal = bootstrap.Modal.getInstance(document.getElementById('taskDetailModal'));
        if (detailModal) detailModal.hide();
        
        // Reload tasks
        loadProjectTasks();
        
        // Show success message
        alert('המשימה נמחקה בהצלחה');
    })
    .catch(error => {
        console.error('Error deleting task:', error);
        alert('שגיאה במחיקת המשימה: ' + error.message);
    });
}

// Function to edit task
function editTask(task) {
    // Close task detail modal
    const detailModal = bootstrap.Modal.getInstance(document.getElementById('taskDetailModal'));
    if (detailModal) detailModal.hide();
    
    // Get form elements
    const idElement = document.getElementById('editTaskId');
    const titleElement = document.getElementById('editTaskTitle');
    const descriptionElement = document.getElementById('editTaskDescription');
    const statusElement = document.getElementById('editTaskStatus');
    const assigneeElement = document.getElementById('editTaskAssignee');
    const dueDateElement = document.getElementById('editTaskDueDate');
    const priorityElement = document.getElementById('editTaskPriority');
    
    // Check if form elements exist
    if (!idElement || !titleElement || !statusElement || !priorityElement) {
        console.error('Edit task form elements not found');
        return;
    }
    
    // Populate form with task data
    idElement.value = task._id;
    titleElement.value = task.title;
    if (descriptionElement) descriptionElement.value = task.description || '';
    statusElement.value = task.status;
    if (dueDateElement) dueDateElement.value = task.due_date ? task.due_date.substring(0, 10) : '';
    priorityElement.value = task.priority;
    
    // Load project members for assignee dropdown
    if (assigneeElement) {
        assigneeElement.innerHTML = '<option value="">-- לא הוקצה --</option>';
        
        fetch(`/api/projects/${task.project_id}/members`)
            .then(response => response.json())
            .then(data => {
                // Add project owner and members to dropdown
                let members = data.members || [];
                if (data.owner) {
                    members.push({username: data.owner, role: 'owner'});
                }
                
                members.forEach(member => {
                    const option = document.createElement('option');
                    option.value = member.username;
                    option.textContent = member.username;
                    if (member.username === task.assigned_to) {
                        option.selected = true;
                    }
                    assigneeElement.appendChild(option);
                });
                
                // Open edit modal
                const editModal = new bootstrap.Modal(document.getElementById('editTaskModal'));
                if (editModal) editModal.show();
                
                // Setup save button event listener
                setupSaveTaskEditBtn(task.project_id);
            })
            .catch(error => {
                console.error('Error loading project members:', error);
                alert('שגיאה בטעינת חברי הפרויקט: ' + error.message);
            });
    } else {
        // Open edit modal without loading members
        const editModal = new bootstrap.Modal(document.getElementById('editTaskModal'));
        if (editModal) editModal.show();
        
        // Setup save button event listener
        setupSaveTaskEditBtn(task.project_id);
    }
}

// Setup save task edit button event listener
function setupSaveTaskEditBtn(projectId) {
    const saveBtn = document.getElementById('saveTaskEditBtn');
    if (!saveBtn) {
        console.error('Save task edit button not found');
        return;
    }
    
    // Remove previous event listeners by cloning
    const newSaveBtn = saveBtn.cloneNode(true);
    saveBtn.parentNode.replaceChild(newSaveBtn, saveBtn);
    
    newSaveBtn.addEventListener('click', function() {
        saveTaskEdit(projectId);
    });
}

// Function to save task edits
function saveTaskEdit(projectId) {
    // Get form values
    const idElement = document.getElementById('editTaskId');
    const titleElement = document.getElementById('editTaskTitle');
    const descriptionElement = document.getElementById('editTaskDescription');
    const statusElement = document.getElementById('editTaskStatus');
    const assigneeElement = document.getElementById('editTaskAssignee');
    const dueDateElement = document.getElementById('editTaskDueDate');
    const priorityElement = document.getElementById('editTaskPriority');
    
    // Check if required elements exist
    if (!idElement || !titleElement || !statusElement || !priorityElement) {
        console.error('Edit task form elements not found');
        return;
    }
    
    const taskId = idElement.value;
    const title = titleElement.value;
    const description = descriptionElement ? descriptionElement.value : '';
    const status = statusElement.value;
    const assignee = assigneeElement ? assigneeElement.value : '';
    const dueDate = dueDateElement ? dueDateElement.value : '';
    const priority = priorityElement.value;
    
    if (!title) {
        alert('כותרת המשימה חובה');
        return;
    }
    
    const taskData = {
        title: title,
        description: description,
        status: status,
        assigned_to: assignee,
        due_date: dueDate,
        priority: priority
    };
    
    fetch(`/api/tasks/${taskId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('שגיאה בעדכון המשימה');
        }
        return response.json();
    })
    .then(data => {
        // Close modal
        const editModal = bootstrap.Modal.getInstance(document.getElementById('editTaskModal'));
        if (editModal) editModal.hide();
        
        // Reload tasks
        loadProjectTasks();
        
        // Show success message
        alert('המשימה עודכנה בהצלחה');
    })
    .catch(error => {
        console.error('Error updating task:', error);
        alert('שגיאה בעדכון המשימה: ' + error.message);
    });
}

// Function to load project members for the assignee dropdown
function loadProjectMembers() {
    const projectId = projectIdValue; // This should be set by the template
    const assigneeSelect = document.getElementById('taskAssignee');
    
    if (!assigneeSelect) {
        console.error('Task assignee select element not found');
        return;
    }
    
    assigneeSelect.innerHTML = '<option value="">-- לא הוקצה --</option>';
    
    fetch(`/api/projects/${projectId}/members`)
        .then(response => response.json())
        .then(data => {
            // Add project owner and members to dropdown
            let members = data.members || [];
            if (data.owner) {
                members.push({username: data.owner, role: 'owner'});
            }
            
            members.forEach(member => {
                const option = document.createElement('option');
                option.value = member.username;
                option.textContent = member.username;
                assigneeSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error loading project members:', error);
        });
}

// Initialize task management functionality
function initTaskManagement() {
    // Set up event listeners
    const tasksTab = document.getElementById('tasks-tab');
    if (tasksTab) {
        tasksTab.addEventListener('click', loadProjectTasks);
    }
    
    const createTaskBtn = document.getElementById('createTaskBtn');
    if (createTaskBtn) {
        createTaskBtn.addEventListener('click', createTask);
    }
    
    const addTaskModal = document.getElementById('addTaskModal');
    if (addTaskModal) {
        addTaskModal.addEventListener('show.bs.modal', loadProjectMembers);
    }
    
    // Load tasks if tasks tab is active on page load
    if (tasksTab && tasksTab.classList.contains('active')) {
        loadProjectTasks();
    }
}

// Call initialize function when the document is loaded
document.addEventListener('DOMContentLoaded', initTaskManagement); 