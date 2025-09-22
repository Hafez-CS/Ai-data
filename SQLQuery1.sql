use master
go

CREATE DATABASE AI
COLLATE Persian_100_CI_AI;
GO

use AI
GO





CREATE TABLE dbo.ai (
    id INT IDENTITY(1,1) PRIMARY KEY,
    file_path NVARCHAR(MAX) NOT NULL, -- محتوای فایل به صورت base64
    chat_id INT NOT NULL,
    filename NVARCHAR(255),
    file_type NVARCHAR(50),
    user_id INT NULL,
    status NVARCHAR(50) DEFAULT 'pending',
    created_at DATETIME DEFAULT GETDATE()
);

CREATE TABLE dbo.result_table (
    id INT IDENTITY(1,1) PRIMARY KEY,
    chat_id INT NOT NULL,
    ai_id INT,
    result_json NVARCHAR(MAX) NOT NULL, -- خروجی JSON
    model_used NVARCHAR(100),
    target_column NVARCHAR(100),
    plot_data NVARCHAR(MAX), -- نمودار به صورت base64
    status NVARCHAR(50) DEFAULT 'completed',
    created_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (ai_id) REFERENCES dbo.ai(id)
);



-- ایندکس روی chat_id برای دسترسی سریع به رکوردهای یک کاربر/چت
CREATE INDEX IX_ai_chatid ON dbo.ai(chat_id);

-- ایندکس روی created_at برای مرتب‌سازی زمانی
CREATE INDEX IX_ai_created_at ON dbo.ai(created_at);

-- ایندکس روی ai_id برای join سریع بین result_table و ai
CREATE INDEX IX_result_aiid ON dbo.result_table(ai_id);

-- ایندکس روی chat_id در جدول نتیجه‌ها
CREATE INDEX IX_result_chatid ON dbo.result_table(chat_id);

SELECT r.id, r.result_json, r.model_used, a.filename, a.file_type
FROM dbo.result_table r
INNER JOIN dbo.ai a ON r.ai_id = a.id
WHERE r.chat_id = 12345
ORDER BY r.created_at DESC;



CREATE PARTITION FUNCTION pfRange (datetime)
AS RANGE RIGHT FOR VALUES ('2025-01-01', '2030-01-01', '2040-01-01');
