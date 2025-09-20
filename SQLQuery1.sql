use master
go

create database AI;
use AI
GO




CREATE TABLE dbo.ai (
    id INT IDENTITY(1,1) PRIMARY KEY,
    file_path NVARCHAR(MAX) NOT NULL, -- مسیر یا محتوای فایل
    created_at DATETIME DEFAULT GETDATE()
);

CREATE TABLE dbo.result_table (
    id INT IDENTITY(1,1) PRIMARY KEY,
    chat_id INT NOT NULL,
    result_json NVARCHAR(MAX) NOT NULL, -- خروجی JSON
    created_at DATETIME DEFAULT GETDATE()
);




